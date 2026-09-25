#include "splat_render/renderer.hpp"

#include "gut_spv.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

#if SPLAT_DRENDER_HAS_VULKAN
#include "splat_drender/vulkan_api.h"
#endif

namespace splat_render {
namespace {

void check(const VkResult result, const char* what) {
    if (result == VK_SUCCESS) return;
    throw std::runtime_error(std::string(what) + " failed (" + std::to_string(result) + ")");
}

VkShaderModule shader_module(VkDevice device, const std::uint32_t* words, std::size_t count) {
    VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize = count * sizeof(std::uint32_t);
    info.pCode = words;
    VkShaderModule module{};
    check(vkCreateShaderModule(device, &info, nullptr, &module), "vkCreateShaderModule");
    return module;
}

float activated_opacity(const float logit) {
    if (logit >= 0.F) {
        const float z = std::exp(-logit);
        return 1.F / (1.F + z);
    }
    const float z = std::exp(std::max(logit, -80.F));
    return z / (1.F + z);
}

float activated_scale(const float log_scale) {
    return std::exp(std::clamp(log_scale, -20.F, 20.F));
}

struct ActivatedGaussian {
    float scales[3]{};
    float rotation[4]{};
    float opacity{};
    float opacity_factor{1.F};
};

ActivatedGaussian activate_gaussian(
    const GaussianCloud& cloud, const std::size_t index) {
    ActivatedGaussian activated;
    const float filter = cloud.filter_3d == nullptr ? 0.F : cloud.filter_3d[index];
    const float filter_squared = filter * filter;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        const float raw = activated_scale(cloud.log_scales[index * 3U + axis]);
        const float filtered = std::sqrt(raw * raw + filter_squared);
        activated.scales[axis] = filtered;
        activated.opacity_factor *= raw / filtered;
    }

    float norm_squared = 0.F;
    for (std::size_t component = 0; component < 4; ++component) {
        const float value = cloud.quaternions[index * 4U + component];
        activated.rotation[component] = value;
        norm_squared += value * value;
    }
    const float inverse_norm =
        1.F / std::sqrt(std::max(norm_squared, 1e-20F));
    for (float& component : activated.rotation) component *= inverse_norm;
    activated.opacity =
        activated_opacity(cloud.opacity_logits[index]) * activated.opacity_factor;
    return activated;
}

struct Push {
    std::uint32_t count{};
    std::uint32_t groups{};
    std::uint32_t shift{};
    std::uint32_t parity{};
};

void memory_barrier(
    VkCommandBuffer command, VkPipelineStageFlags src, VkPipelineStageFlags dst) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask =
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(command, src, dst, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

}  // namespace

#if SPLAT_DRENDER_HAS_VULKAN
struct Renderer::EwaPreview {
    EwaPreview(const splat_drender::vulkan::ContextOptions& options, const bool shared)
        : context(options), shared_device(shared) {}

    splat_drender::vulkan::Context context;
    splat_drender::vulkan::SplatRasterizer rasterizer{context};
    // True when the context adopted the editor's device, so the frame can stay
    // in VRAM and be copied into the viewport image without a host round trip.
    bool shared_device = false;
};
#endif

Renderer::~Renderer() { reset(); }

void Renderer::attach(const Device& device) {
    if (device_ == device.device && device_ != VK_NULL_HANDLE) {
        physical_ = device.physical;
        queue_ = device.queue;
        family_ = device.queue_family;
        return;
    }
    reset();
    physical_ = device.physical;
    device_ = device.device;
    queue_ = device.queue;
    family_ = device.queue_family;
    supported_ = true;
    failure_.clear();
}

void Renderer::reset() {
#if SPLAT_DRENDER_HAS_VULKAN
    delete ewa_;
#endif
    ewa_ = nullptr;
    ewa_opacity_factors_.clear();
    if (device_ == VK_NULL_HANDLE) {
        ready_ = false;
        count_ = 0;
        display_ = -1;
        cache_valid_ = false;
        frame_mapped_ = nullptr;
        return;
    }
    vkDeviceWaitIdle(device_);
    destroy_frames();
    destroy_storage(centers_);
    destroy_storage(scales_);
    destroy_storage(rotations_);
    destroy_storage(harmonics_);
    destroy_storage(quads_);
    destroy_storage(keys0_);
    destroy_storage(keys1_);
    destroy_storage(vals0_);
    destroy_storage(vals1_);
    destroy_storage(wg_counts_);
    destroy_storage(wg_offsets_);
    destroy_storage(ranks_);
    destroy_storage(totals_);
    destroy_storage(bin_start_);
    destroy_storage(dummy_);
    destroy_storage(rgba_staging_);
    if (frame_mapped_) {
        vkUnmapMemory(device_, frame_ubo_.memory);
        frame_mapped_ = nullptr;
    }
    destroy_storage(frame_ubo_);
    destroy_device_objects();
    physical_ = {};
    device_ = {};
    queue_ = {};
    family_ = VK_QUEUE_FAMILY_IGNORED;
    ready_ = false;
    count_ = 0;
    source_key_.clear();
    cache_valid_ = false;
    display_ = -1;
}

void Renderer::destroy_storage(Storage& storage) {
    if (device_ == VK_NULL_HANDLE) {
        storage = {};
        return;
    }
    if (storage.buffer) vkDestroyBuffer(device_, storage.buffer, nullptr);
    if (storage.memory) vkFreeMemory(device_, storage.memory, nullptr);
    storage = {};
}

void Renderer::destroy_frames() {
    if (device_ == VK_NULL_HANDLE) {
        for (Frame& frame : frames_) frame = {};
        display_ = -1;
        write_ = 0;
        return;
    }
    for (Frame& frame : frames_) {
        if (frame.sampler) vkDestroySampler(device_, frame.sampler, nullptr);
        if (frame.framebuffer) vkDestroyFramebuffer(device_, frame.framebuffer, nullptr);
        if (frame.view) vkDestroyImageView(device_, frame.view, nullptr);
        if (frame.color) vkDestroyImage(device_, frame.color, nullptr);
        if (frame.memory) vkFreeMemory(device_, frame.memory, nullptr);
        frame = {};
    }
    display_ = -1;
    write_ = 0;
    cache_valid_ = false;
    ++frames_epoch_;
}

void Renderer::destroy_device_objects() {
    if (device_ == VK_NULL_HANDLE) return;
    if (keys_pipeline_) vkDestroyPipeline(device_, keys_pipeline_, nullptr);
    if (hist_pipeline_) vkDestroyPipeline(device_, hist_pipeline_, nullptr);
    if (scan_pipeline_) vkDestroyPipeline(device_, scan_pipeline_, nullptr);
    if (scatter_pipeline_) vkDestroyPipeline(device_, scatter_pipeline_, nullptr);
    if (ring_prepare_pipeline_) vkDestroyPipeline(device_, ring_prepare_pipeline_, nullptr);
    if (ring_pipeline_) vkDestroyPipeline(device_, ring_pipeline_, nullptr);
    if (render_pass_) vkDestroyRenderPass(device_, render_pass_, nullptr);
    if (pipeline_layout_) vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    if (set_layout_) vkDestroyDescriptorSetLayout(device_, set_layout_, nullptr);
    if (pool_) vkDestroyDescriptorPool(device_, pool_, nullptr);
    if (fence_) vkDestroyFence(device_, fence_, nullptr);
    if (command_pool_) vkDestroyCommandPool(device_, command_pool_, nullptr);
    keys_pipeline_ = {};
    hist_pipeline_ = {};
    scan_pipeline_ = {};
    scatter_pipeline_ = {};
    ring_prepare_pipeline_ = {};
    ring_pipeline_ = {};
    render_pass_ = {};
    pipeline_layout_ = {};
    set_layout_ = {};
    pool_ = {};
    set_ = {};
    fence_ = {};
    command_pool_ = {};
    command_ = {};
}

std::uint32_t Renderer::memory_type(
    const std::uint32_t bits, const VkMemoryPropertyFlags flags) const {
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_, &properties);
    for (std::uint32_t index = 0; index < properties.memoryTypeCount; ++index) {
        if ((bits & (1U << index)) &&
            (properties.memoryTypes[index].propertyFlags & flags) == flags)
            return index;
    }
    throw std::runtime_error("No matching Vulkan memory type");
}

void Renderer::create_storage(
    Storage& storage, VkDeviceSize size, const VkBufferUsageFlags usage,
    const VkMemoryPropertyFlags memory) {
    destroy_storage(storage);
    if (size == 0) size = 4;
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    info.size = size;
    info.usage = usage;
    check(vkCreateBuffer(device_, &info, nullptr, &storage.buffer), "vkCreateBuffer");
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, storage.buffer, &requirements);
    VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = memory_type(requirements.memoryTypeBits, memory);
    check(vkAllocateMemory(device_, &allocation, nullptr, &storage.memory), "vkAllocateMemory");
    check(vkBindBufferMemory(device_, storage.buffer, storage.memory, 0), "vkBindBufferMemory");
    storage.size = size;
}

void Renderer::wait_gpu() {
    if (fence_)
        check(vkWaitForFences(device_, 1, &fence_, VK_TRUE, UINT64_MAX), "vkWaitForFences");
}

void Renderer::upload_storage(
    Storage& storage, const void* data, const VkDeviceSize size) {
    create_storage(
        storage, size,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (data == nullptr || size == 0) return;
    Storage staging{};
    create_storage(
        staging, size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* mapped{};
    check(vkMapMemory(device_, staging.memory, 0, size, 0, &mapped), "vkMapMemory");
    std::memcpy(mapped, data, static_cast<std::size_t>(size));
    vkUnmapMemory(device_, staging.memory);

    check(vkResetCommandBuffer(command_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command_, &begin), "vkBeginCommandBuffer");
    VkBufferCopy copy{};
    copy.size = size;
    vkCmdCopyBuffer(command_, staging.buffer, storage.buffer, 1, &copy);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = storage.buffer;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(
        command_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &barrier, 0, nullptr);
    check(vkEndCommandBuffer(command_), "vkEndCommandBuffer");
    check(vkResetFences(device_, 1, &fence_), "vkResetFences");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_;
    check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");
    wait_gpu();
    destroy_storage(staging);
}

void Renderer::write_descriptors() {
    const Storage bindings[] = {
        frame_ubo_, centers_, scales_, rotations_, harmonics_, quads_,
        keys0_, keys1_, vals0_, vals1_, wg_counts_, wg_offsets_, ranks_,
        totals_, bin_start_};
    VkDescriptorBufferInfo infos[15]{};
    VkWriteDescriptorSet writes[15]{};
    for (std::uint32_t index = 0; index < 15; ++index) {
        const Storage& storage = bindings[index].buffer ? bindings[index] : dummy_;
        infos[index].buffer = storage.buffer;
        infos[index].offset = 0;
        infos[index].range = storage.size;
        writes[index].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[index].dstSet = set_;
        writes[index].dstBinding = index;
        writes[index].descriptorCount = 1;
        writes[index].descriptorType = index == 0
            ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[index].pBufferInfo = &infos[index];
    }
    vkUpdateDescriptorSets(device_, 15, writes, 0, nullptr);
}

bool Renderer::ensure_device() {
    if (device_ == VK_NULL_HANDLE) {
        failure_ = "Vulkan device is not attached";
        return false;
    }
    if (ready_) return true;
    if (!supported_) return false;

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, families.data());
    if (family_ >= family_count ||
        (families[family_].queueFlags & VK_QUEUE_COMPUTE_BIT) == 0 ||
        (families[family_].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) {
        supported_ = false;
        failure_ = "The Vulkan queue cannot run compute and graphics";
        return false;
    }

    VkDescriptorSetLayoutBinding layout_bindings[15]{};
    for (std::uint32_t index = 0; index < 15; ++index) {
        layout_bindings[index].binding = index;
        layout_bindings[index].descriptorType = index == 0
            ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
            : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layout_bindings[index].descriptorCount = 1;
        layout_bindings[index].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT |
                                             VK_SHADER_STAGE_VERTEX_BIT |
                                             VK_SHADER_STAGE_FRAGMENT_BIT;
    }
    VkDescriptorSetLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    layout_info.bindingCount = 15;
    layout_info.pBindings = layout_bindings;
    check(vkCreateDescriptorSetLayout(device_, &layout_info, nullptr, &set_layout_),
          "vkCreateDescriptorSetLayout");

    VkPushConstantRange push{};
    push.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push.offset = 0;
    push.size = sizeof(Push);
    VkPipelineLayoutCreateInfo pipeline_layout{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pipeline_layout.setLayoutCount = 1;
    pipeline_layout.pSetLayouts = &set_layout_;
    pipeline_layout.pushConstantRangeCount = 1;
    pipeline_layout.pPushConstantRanges = &push;
    check(vkCreatePipelineLayout(device_, &pipeline_layout, nullptr, &pipeline_layout_),
          "vkCreatePipelineLayout");

    const VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 14},
    };
    VkDescriptorPoolCreateInfo pool_info{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 2;
    pool_info.pPoolSizes = pool_sizes;
    check(vkCreateDescriptorPool(device_, &pool_info, nullptr, &pool_),
          "vkCreateDescriptorPool");
    VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    allocate.descriptorPool = pool_;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &set_layout_;
    check(vkAllocateDescriptorSets(device_, &allocate, &set_), "vkAllocateDescriptorSets");

    VkAttachmentDescription color{};
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    VkAttachmentReference color_ref{0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL};
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_ref;
    VkSubpassDependency dependencies[2]{};
    dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[0].dstSubpass = 0;
    dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].srcSubpass = 0;
    dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    VkRenderPassCreateInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO};
    pass.attachmentCount = 1;
    pass.pAttachments = &color;
    pass.subpassCount = 1;
    pass.pSubpasses = &subpass;
    pass.dependencyCount = 2;
    pass.pDependencies = dependencies;
    check(vkCreateRenderPass(device_, &pass, nullptr, &render_pass_), "vkCreateRenderPass");

    const auto make_compute = [&](const std::uint32_t* words, std::size_t count) {
        const VkShaderModule module = shader_module(device_, words, count);
        VkPipelineShaderStageCreateInfo stage{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = module;
        stage.pName = "main";
        VkComputePipelineCreateInfo info{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        info.stage = stage;
        info.layout = pipeline_layout_;
        VkPipeline pipeline{};
        check(vkCreateComputePipelines(device_, {}, 1, &info, nullptr, &pipeline),
              "vkCreateComputePipelines");
        vkDestroyShaderModule(device_, module, nullptr);
        return pipeline;
    };
    keys_pipeline_ = make_compute(gut_spv::gut_keys_comp, std::size(gut_spv::gut_keys_comp));
    hist_pipeline_ = make_compute(gut_spv::gut_hist_comp, std::size(gut_spv::gut_hist_comp));
    scan_pipeline_ = make_compute(gut_spv::gut_scan_comp, std::size(gut_spv::gut_scan_comp));
    scatter_pipeline_ = make_compute(
        gut_spv::gut_scatter_comp, std::size(gut_spv::gut_scatter_comp));
    ring_prepare_pipeline_ = make_compute(
        gut_spv::ring_prepare_cs_hlsl, std::size(gut_spv::ring_prepare_cs_hlsl));

    const auto make_graphics = [&](const std::uint32_t* vert_words, std::size_t vert_count,
                                   const std::uint32_t* frag_words, std::size_t frag_count) {
        const VkShaderModule vert = shader_module(device_, vert_words, vert_count);
        const VkShaderModule frag = shader_module(device_, frag_words, frag_count);
        VkPipelineShaderStageCreateInfo stages[2]{};
        stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
        stages[0].module = vert;
        stages[0].pName = "main";
        stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        stages[1].module = frag;
        stages[1].pName = "main";
        VkPipelineVertexInputStateCreateInfo vertex{
            VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        VkPipelineInputAssemblyStateCreateInfo assembly{
            VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        VkPipelineViewportStateCreateInfo viewport{
            VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        viewport.viewportCount = 1;
        viewport.scissorCount = 1;
        VkPipelineRasterizationStateCreateInfo raster{
            VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        raster.polygonMode = VK_POLYGON_MODE_FILL;
        raster.cullMode = VK_CULL_MODE_NONE;
        raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        raster.lineWidth = 1.F;
        VkPipelineMultisampleStateCreateInfo multisample{
            VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        VkPipelineColorBlendAttachmentState blend{};
        blend.blendEnable = VK_TRUE;
        blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
        blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.colorBlendOp = VK_BLEND_OP_ADD;
        blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
        blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
        blend.alphaBlendOp = VK_BLEND_OP_ADD;
        blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                               VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo blending{
            VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        blending.attachmentCount = 1;
        blending.pAttachments = &blend;
        const VkDynamicState dynamic_states[] = {
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic{
            VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        dynamic.dynamicStateCount = 2;
        dynamic.pDynamicStates = dynamic_states;
        VkGraphicsPipelineCreateInfo graphics{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
        graphics.stageCount = 2;
        graphics.pStages = stages;
        graphics.pVertexInputState = &vertex;
        graphics.pInputAssemblyState = &assembly;
        graphics.pViewportState = &viewport;
        graphics.pRasterizationState = &raster;
        graphics.pMultisampleState = &multisample;
        graphics.pColorBlendState = &blending;
        graphics.pDynamicState = &dynamic;
        graphics.layout = pipeline_layout_;
        graphics.renderPass = render_pass_;
        VkPipeline pipeline{};
        check(vkCreateGraphicsPipelines(device_, {}, 1, &graphics, nullptr, &pipeline),
              "vkCreateGraphicsPipelines");
        vkDestroyShaderModule(device_, vert, nullptr);
        vkDestroyShaderModule(device_, frag, nullptr);
        return pipeline;
    };
    ring_pipeline_ = make_graphics(
        gut_spv::ring_draw_vs_hlsl, std::size(gut_spv::ring_draw_vs_hlsl),
        gut_spv::ring_draw_ps_hlsl, std::size(gut_spv::ring_draw_ps_hlsl));

    VkCommandPoolCreateInfo command_pool{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    command_pool.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool.queueFamilyIndex = family_;
    check(vkCreateCommandPool(device_, &command_pool, nullptr, &command_pool_),
          "vkCreateCommandPool");
    VkCommandBufferAllocateInfo command_info{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    command_info.commandPool = command_pool_;
    command_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_info.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(device_, &command_info, &command_),
          "vkAllocateCommandBuffers");
    VkFenceCreateInfo fence{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    check(vkCreateFence(device_, &fence, nullptr, &fence_), "vkCreateFence");

    create_storage(
        frame_ubo_, 256, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkMapMemory(device_, frame_ubo_.memory, 0, 256, 0, &frame_mapped_), "vkMapMemory");
    create_storage(
        dummy_, 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    create_storage(
        totals_, sizeof(std::uint32_t) * 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    create_storage(
        bin_start_, sizeof(std::uint32_t) * 256, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    write_descriptors();
    ready_ = true;
    failure_.clear();
    return true;
}

void Renderer::clear_model() {
    if (device_ != VK_NULL_HANDLE && ready_) {
        wait_gpu();
        destroy_storage(centers_);
        destroy_storage(scales_);
        destroy_storage(rotations_);
        destroy_storage(harmonics_);
        destroy_storage(quads_);
        destroy_storage(keys0_);
        destroy_storage(keys1_);
        destroy_storage(vals0_);
        destroy_storage(vals1_);
        destroy_storage(wg_counts_);
        destroy_storage(wg_offsets_);
        destroy_storage(ranks_);
        if (set_) write_descriptors();
    }
    count_ = 0;
    source_key_.clear();
    cache_valid_ = false;
    ewa_opacity_factors_.clear();
    ++generation_;
#if SPLAT_DRENDER_HAS_VULKAN
    if (ewa_) ewa_->rasterizer.clear_model();
#endif
}

bool Renderer::upload(const GaussianCloud& cloud, const std::string_view source_key) {
    failure_.clear();
    if (!ensure_device()) return false;
    if (!source_key.empty() && source_key == source_key_ && count_ == cloud.count)
        return true;
    if (cloud.count > 16'000'000U) {
        failure_ = "Too many Gaussians for the Vulkan preview";
        return false;
    }
    if (cloud.count > 0 &&
        (cloud.means == nullptr || cloud.log_scales == nullptr ||
         cloud.quaternions == nullptr || cloud.opacity_logits == nullptr)) {
        failure_ = "Gaussian upload is missing attribute arrays";
        return false;
    }

    std::uint32_t bases = std::max(1U, cloud.sh_bases);
    std::uint32_t degree = std::min(cloud.sh_degree, 3U);
    while (degree > 0 && (degree + 1U) * (degree + 1U) > bases) --degree;
    if (cloud.sh == nullptr) {
        bases = 1;
        degree = 0;
    }
    const std::size_t count = cloud.count;
    std::vector<float> centers(count * 4U);
    std::vector<float> scales(count * 4U, 1.F);
    std::vector<float> rotations(count * 4U);
    std::vector<float> harmonics(count * bases * 3U, 0.F);
    for (std::size_t index = 0; index < count; ++index) {
        const ActivatedGaussian activated = activate_gaussian(cloud, index);
        centers[index * 4U] = cloud.means[index * 3U];
        centers[index * 4U + 1U] = cloud.means[index * 3U + 1U];
        centers[index * 4U + 2U] = cloud.means[index * 3U + 2U];
        centers[index * 4U + 3U] = activated.opacity;
        scales[index * 4U] = activated.scales[0];
        scales[index * 4U + 1U] = activated.scales[1];
        scales[index * 4U + 2U] = activated.scales[2];
        rotations[index * 4U] = activated.rotation[0];
        rotations[index * 4U + 1U] = activated.rotation[1];
        rotations[index * 4U + 2U] = activated.rotation[2];
        rotations[index * 4U + 3U] = activated.rotation[3];
        if (cloud.sh != nullptr) {
            const std::uint32_t src_bases = std::max(cloud.sh_bases, bases);
            const std::size_t src = index * src_bases * 3U;
            const std::size_t dst = index * bases * 3U;
            std::memcpy(
                harmonics.data() + dst, cloud.sh + src,
                static_cast<std::size_t>(bases) * 3U * sizeof(float));
        }
    }

    wait_gpu();
    const std::uint32_t groups = count == 0 ? 1U : static_cast<std::uint32_t>((count + 255U) / 256U);
    const VkDeviceSize sort_bytes = std::max<VkDeviceSize>(count, 1) * sizeof(std::uint32_t);
    const VkDeviceSize group_bytes =
        static_cast<VkDeviceSize>(groups) * 256U * sizeof(std::uint32_t);
    upload_storage(centers_, centers.data(), centers.size() * sizeof(float));
    upload_storage(scales_, scales.data(), scales.size() * sizeof(float));
    upload_storage(rotations_, rotations.data(), rotations.size() * sizeof(float));
    upload_storage(harmonics_, harmonics.data(), harmonics.size() * sizeof(float));
    create_storage(
        quads_, std::max<VkDeviceSize>(count, 1) * 7U * sizeof(float) * 4U,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    create_storage(
        keys0_, sort_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    create_storage(
        keys1_, sort_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    create_storage(
        vals0_, sort_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    create_storage(
        vals1_, sort_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    create_storage(
        ranks_, sort_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    create_storage(
        wg_counts_, group_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    create_storage(
        wg_offsets_, group_bytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    write_descriptors();

    count_ = cloud.count;
    sh_degree_ = degree;
    sh_bases_ = bases;
    source_key_ = std::string(source_key);
    cache_valid_ = false;
    ++generation_;
    failure_.clear();
    try {
        sync_ewa(cloud);
    } catch (const std::exception& error) {
        failure_ = error.what();
        return false;
    }
    return true;
}

bool Renderer::update_centers(const float* centers, const std::uint32_t count) {
    if (!ready_ || centers == nullptr || count != count_ || count_ == 0) return false;
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(count) * sizeof(float) * 4U;
    if (!centers_.buffer || centers_.size < bytes) return false;
    Storage staging{};
    create_storage(
        staging, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    void* mapped{};
    check(vkMapMemory(device_, staging.memory, 0, bytes, 0, &mapped), "vkMapMemory");
    std::vector<float> activated_centers(
        centers, centers + static_cast<std::size_t>(count) * 4U);
    for (std::uint32_t index = 0; index < count; ++index) {
        const float factor = index < ewa_opacity_factors_.size()
            ? ewa_opacity_factors_[index]
            : 1.F;
        activated_centers[static_cast<std::size_t>(index) * 4U + 3U] *= factor;
    }
    std::memcpy(
        mapped, activated_centers.data(), static_cast<std::size_t>(bytes));
    vkUnmapMemory(device_, staging.memory);
    wait_gpu();
    check(vkResetFences(device_, 1, &fence_), "vkResetFences");
    check(vkResetCommandBuffer(command_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command_, &begin), "vkBeginCommandBuffer");
    VkBufferCopy copy{};
    copy.size = bytes;
    vkCmdCopyBuffer(command_, staging.buffer, centers_.buffer, 1, &copy);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = centers_.buffer;
    barrier.size = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(
        command_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &barrier, 0, nullptr);
    check(vkEndCommandBuffer(command_), "vkEndCommandBuffer");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_;
    check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");
    wait_gpu();
    destroy_storage(staging);
    cache_valid_ = false;
    ++generation_;
#if SPLAT_DRENDER_HAS_VULKAN
    if (ewa_ && ewa_->rasterizer.has_model()) {
        std::vector<float> means(static_cast<std::size_t>(count) * 3U);
        std::vector<float> opacities(count);
        for (std::uint32_t index = 0; index < count; ++index) {
            means[static_cast<std::size_t>(index) * 3U] = centers[index * 4U];
            means[static_cast<std::size_t>(index) * 3U + 1U] = centers[index * 4U + 1U];
            means[static_cast<std::size_t>(index) * 3U + 2U] = centers[index * 4U + 2U];
            opacities[index] =
                activated_centers[static_cast<std::size_t>(index) * 4U + 3U];
        }
        try {
            ewa_->rasterizer.update_means_and_opacities(means, opacities);
        } catch (const std::exception& error) {
            failure_ = error.what();
            return false;
        }
    }
#endif
    return true;
}

bool Renderer::ensure_frames(const std::uint32_t width, const std::uint32_t height) {
    if (frames_[0].color && frames_[0].width == width && frames_[0].height == height)
        return true;
    wait_gpu();
    destroy_frames();
    for (Frame& frame : frames_) {
        VkImageCreateInfo image{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        image.imageType = VK_IMAGE_TYPE_2D;
        image.format = VK_FORMAT_R8G8B8A8_UNORM;
        image.extent = {width, height, 1};
        image.mipLevels = 1;
        image.arrayLayers = 1;
        image.samples = VK_SAMPLE_COUNT_1_BIT;
        image.tiling = VK_IMAGE_TILING_OPTIMAL;
        image.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        check(vkCreateImage(device_, &image, nullptr, &frame.color), "vkCreateImage");
        VkMemoryRequirements requirements{};
        vkGetImageMemoryRequirements(device_, frame.color, &requirements);
        VkMemoryAllocateInfo allocation{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        allocation.allocationSize = requirements.size;
        allocation.memoryTypeIndex = memory_type(
            requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(device_, &allocation, nullptr, &frame.memory), "vkAllocateMemory");
        check(vkBindImageMemory(device_, frame.color, frame.memory, 0), "vkBindImageMemory");
        VkImageViewCreateInfo view{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        view.image = frame.color;
        view.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view.format = VK_FORMAT_R8G8B8A8_UNORM;
        view.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        check(vkCreateImageView(device_, &view, nullptr, &frame.view), "vkCreateImageView");
        VkFramebufferCreateInfo framebuffer{VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO};
        framebuffer.renderPass = render_pass_;
        framebuffer.attachmentCount = 1;
        framebuffer.pAttachments = &frame.view;
        framebuffer.width = width;
        framebuffer.height = height;
        framebuffer.layers = 1;
        check(vkCreateFramebuffer(device_, &framebuffer, nullptr, &frame.framebuffer),
              "vkCreateFramebuffer");
        VkSamplerCreateInfo sampler{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
        sampler.magFilter = VK_FILTER_LINEAR;
        sampler.minFilter = VK_FILTER_LINEAR;
        sampler.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        sampler.maxLod = 1.F;
        check(vkCreateSampler(device_, &sampler, nullptr, &frame.sampler), "vkCreateSampler");
        frame.width = width;
        frame.height = height;
    }
    return true;
}

bool Renderer::draw(const Camera& camera, FrameTarget& target, const Shading shading) {
    target = {};
    if (!ensure_device()) return false;
    static_assert(sizeof(FrameData) == 144);
    static_assert(offsetof(FrameData, sh_degree) == 128);
    const std::uint32_t width = std::max(1U, camera.width);
    const std::uint32_t height = std::max(1U, camera.height);
    FrameData frame{};
    std::memcpy(frame.world_to_camera, camera.world_to_camera.data(), sizeof(frame.world_to_camera));
    frame.eye[0] = camera.position[0];
    frame.eye[1] = camera.position[1];
    frame.eye[2] = camera.position[2];
    frame.eye[3] = static_cast<float>(camera.model);
    frame.intrinsics[0] = camera.fx;
    frame.intrinsics[1] = camera.fy;
    frame.intrinsics[2] = camera.cx;
    frame.intrinsics[3] = camera.cy;
    frame.distortion[0] = camera.k1;
    frame.distortion[1] = camera.k2;
    frame.distortion[2] = camera.k3;
    frame.distortion[3] = camera.k4;
    frame.viewport[0] = static_cast<float>(width);
    frame.viewport[1] = static_cast<float>(height);
    frame.viewport[2] = shading == Shading::rings
        ? (camera.ring_sigma > 0.F ? camera.ring_sigma : 2.828427F)
        : 0.F;
    frame.sh_degree = sh_degree_;
    frame.count = count_;
    frame.bases = sh_bases_;
    frame.reserved = static_cast<std::uint32_t>(shading);
    if (cache_valid_ && display_ >= 0 && cached_generation_ == generation_ &&
        frames_[display_].width == width && frames_[display_].height == height &&
        std::memcmp(&cached_, &frame, sizeof(frame)) == 0) {
        const Frame& shown = frames_[display_];
        target.slot = display_;
        target.view = shown.view;
        target.sampler = shown.sampler;
        target.width = shown.width;
        target.height = shown.height;
        return true;
    }
    if (!ensure_frames(width, height)) return false;
    if (shading == Shading::gaussian) return draw_ewa(camera, frame, target);

    wait_gpu();
    check(vkResetFences(device_, 1, &fence_), "vkResetFences");
    check(vkResetCommandBuffer(command_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command_, &begin), "vkBeginCommandBuffer");
    std::memcpy(frame_mapped_, &frame, sizeof(frame));

    if (count_ > 0) {
        const std::uint32_t groups = (count_ + 255U) / 256U;
        vkCmdBindDescriptorSets(
            command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &set_,
            0, nullptr);
        const auto dispatch = [&](VkPipeline pipeline, const Push& push,
                                  const std::uint32_t groups_x) {
            vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdPushConstants(
                command_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                &push);
            vkCmdDispatch(command_, groups_x, 1, 1);
            memory_barrier(
                command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
        };
        // push.groups is the histogram workgroup count. The scan shader
        // walks that many groups from a single workgroup.
        Push push{count_, groups, 0, 0};
        dispatch(keys_pipeline_, push, groups);
        for (std::uint32_t byte = 0; byte < 4; ++byte) {
            push.shift = byte * 8U;
            push.parity = byte & 1U;
            dispatch(hist_pipeline_, push, groups);
            dispatch(scan_pipeline_, push, 1);
            dispatch(scatter_pipeline_, push, groups);
        }
        push.shift = 0;
        push.parity = 0;
        dispatch(ring_prepare_pipeline_, push, groups);
        memory_barrier(
            command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_VERTEX_SHADER_BIT);
    }

    Frame& image = frames_[write_];
    VkClearValue clear{};
    clear.color = shading == Shading::rings
        ? VkClearColorValue{{9.F / 255.F, 11.F / 255.F, 16.F / 255.F, 1.F}}
        : VkClearColorValue{{0.F, 0.F, 0.F, 1.F}};
    VkRenderPassBeginInfo render{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};
    render.renderPass = render_pass_;
    render.framebuffer = image.framebuffer;
    render.renderArea.extent = {width, height};
    render.clearValueCount = 1;
    render.pClearValues = &clear;
    vkCmdBeginRenderPass(command_, &render, VK_SUBPASS_CONTENTS_INLINE);
    VkViewport viewport{};
    viewport.width = static_cast<float>(width);
    viewport.height = static_cast<float>(height);
    viewport.maxDepth = 1.F;
    VkRect2D scissor{};
    scissor.extent = {width, height};
    vkCmdSetViewport(command_, 0, 1, &viewport);
    vkCmdSetScissor(command_, 0, 1, &scissor);
    if (count_ > 0) {
        vkCmdBindPipeline(command_, VK_PIPELINE_BIND_POINT_GRAPHICS, ring_pipeline_);
        vkCmdBindDescriptorSets(
            command_, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0, 1, &set_,
            0, nullptr);
        vkCmdDraw(command_, 6, count_, 0, 0);
    }
    vkCmdEndRenderPass(command_);
    check(vkEndCommandBuffer(command_), "vkEndCommandBuffer");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_;
    check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");

    display_ = write_;
    write_ = (write_ + 1) % k_frames;
    image.sampled = true;
    cached_ = frame;
    cached_generation_ = generation_;
    cache_valid_ = true;
    target.slot = display_;
    target.view = image.view;
    target.sampler = image.sampler;
    target.width = width;
    target.height = height;
    return true;
}

void Renderer::sync_ewa(const GaussianCloud& cloud) {
#if SPLAT_DRENDER_HAS_VULKAN
    if (cloud.count == 0) {
        ewa_opacity_factors_.clear();
        if (ewa_) ewa_->rasterizer.clear_model();
        return;
    }
    if (!ewa_) {
        // Prefer the editor's own device. Sharing it avoids a second VkDevice
        // for the preview and lets a later pass hand the image over on the GPU.
        // A queue family without compute cannot host the compute passes, so that
        // case falls back to a private device.
        splat_drender::vulkan::ContextOptions options;
        if (family_ != VK_QUEUE_FAMILY_IGNORED && device_ != VK_NULL_HANDLE) {
            std::uint32_t family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, nullptr);
            if (family_ < family_count) {
                std::vector<VkQueueFamilyProperties> families(family_count);
                vkGetPhysicalDeviceQueueFamilyProperties(physical_, &family_count, families.data());
                if ((families[family_].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
                    options.external_device.instance = VK_NULL_HANDLE;
                    options.external_device.physical_device = physical_;
                    options.external_device.device = device_;
                    options.external_device.queue = queue_;
                    options.external_device.queue_family = family_;
                }
            }
        }
        ewa_ = new EwaPreview(options, options.external_device.valid());
    }
    const std::uint32_t count = cloud.count;
    const std::uint32_t bases = std::max(1U, sh_bases_);
    const std::uint32_t src_bases = cloud.sh == nullptr ? 1U : std::max(cloud.sh_bases, bases);
    std::vector<float> means(static_cast<std::size_t>(count) * 3U);
    std::vector<float> opacities(count);
    std::vector<float> scales(static_cast<std::size_t>(count) * 3U);
    std::vector<float> rotations(static_cast<std::size_t>(count) * 4U);
    std::vector<float> harmonics(static_cast<std::size_t>(count) * bases * 3U, 0.0F);
    ewa_opacity_factors_.resize(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        const ActivatedGaussian activated = activate_gaussian(cloud, index);
        means[static_cast<std::size_t>(index) * 3U] = cloud.means[index * 3U];
        means[static_cast<std::size_t>(index) * 3U + 1U] = cloud.means[index * 3U + 1U];
        means[static_cast<std::size_t>(index) * 3U + 2U] = cloud.means[index * 3U + 2U];
        opacities[index] = activated.opacity;
        ewa_opacity_factors_[index] = activated.opacity_factor;
        scales[static_cast<std::size_t>(index) * 3U] = activated.scales[0];
        scales[static_cast<std::size_t>(index) * 3U + 1U] = activated.scales[1];
        scales[static_cast<std::size_t>(index) * 3U + 2U] = activated.scales[2];
        rotations[static_cast<std::size_t>(index) * 4U] = activated.rotation[0];
        rotations[static_cast<std::size_t>(index) * 4U + 1U] = activated.rotation[1];
        rotations[static_cast<std::size_t>(index) * 4U + 2U] = activated.rotation[2];
        rotations[static_cast<std::size_t>(index) * 4U + 3U] = activated.rotation[3];
        if (cloud.sh == nullptr) continue;
        std::memcpy(
            harmonics.data() + static_cast<std::size_t>(index) * bases * 3U,
            cloud.sh + static_cast<std::size_t>(index) * src_bases * 3U,
            static_cast<std::size_t>(bases) * 3U * sizeof(float));
    }
    splat_drender::vulkan::SplatGaussians gaussians;
    gaussians.means = means;
    gaussians.opacities = opacities;
    gaussians.scales = scales;
    gaussians.rotations = rotations;
    gaussians.sh = harmonics;
    gaussians.sh_degree = cloud.sh == nullptr ? 0U : sh_degree_;
    gaussians.sh_bases = cloud.sh == nullptr ? 1U : bases;
    ewa_->rasterizer.upload_model(gaussians);
#else
    (void)cloud;
#endif
}

bool Renderer::draw_ewa(const Camera& camera, const FrameData& frame, FrameTarget& target) {
    const auto width = static_cast<std::uint32_t>(frame.viewport[0]);
    const auto height = static_cast<std::uint32_t>(frame.viewport[1]);
    std::vector<std::uint8_t> rgba;
    VkBuffer device_rgba = VK_NULL_HANDLE;
    VkDeviceSize device_rgba_bytes = 0;
#if SPLAT_DRENDER_HAS_VULKAN
    if (count_ > 0 && ewa_ && ewa_->rasterizer.has_model()) {
        splat_drender::vulkan::SplatCamera trained;
        trained.width = width;
        trained.height = height;
        trained.fx = camera.fx;
        trained.fy = camera.fy;
        trained.cx = camera.cx;
        trained.cy = camera.cy;
        trained.k1 = camera.k1;
        trained.k2 = camera.k2;
        trained.k3 = camera.k3;
        trained.k4 = camera.k4;
        trained.world_to_camera = std::span<const float>(
            camera.world_to_camera.data(), camera.world_to_camera.size());
        trained.center = std::span<const float>(camera.position.data(), camera.position.size());
        switch (camera.model) {
        case k_camera_orthographic: trained.mode = 4; break;
        case k_camera_fisheye: trained.mode = 1; break;
        case k_camera_equirectangular: trained.mode = 3; break;
        default: trained.mode = 0; break;
        }
        splat_drender::vulkan::SplatSettings settings;
        settings.need_depth = false;
        try {
            if (ewa_->shared_device) {
                // The preview runs on this device, so the frame stays in VRAM
                // and the image copy below reads it directly.
                const splat_drender::vulkan::SplatRgbaImage frame_image =
                    ewa_->rasterizer.render_rgba_device(trained, settings);
                device_rgba = frame_image.buffer;
                device_rgba_bytes = static_cast<VkDeviceSize>(frame_image.bytes);
            } else {
                ewa_->rasterizer.render_rgba(trained, settings, rgba);
            }
        } catch (const std::exception& error) {
            failure_ = error.what();
            return false;
        }
    }
#else
    if (count_ > 0) {
        failure_ = "The trained splat renderer is not linked";
        return false;
    }
#endif
    VkBuffer source = device_rgba;
    VkDeviceSize bytes = device_rgba_bytes;
    if (source == VK_NULL_HANDLE) {
        if (rgba.size() != static_cast<std::size_t>(width) * height * 4U) {
            rgba.assign(static_cast<std::size_t>(width) * height * 4U, 0);
            for (std::size_t pixel = 3; pixel < rgba.size(); pixel += 4) rgba[pixel] = 255;
        }
        bytes = static_cast<VkDeviceSize>(rgba.size());
        if (rgba_staging_.buffer == VK_NULL_HANDLE || rgba_staging_.size < bytes) {
            destroy_storage(rgba_staging_);
            create_storage(
                rgba_staging_, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        void* mapped = nullptr;
        check(vkMapMemory(device_, rgba_staging_.memory, 0, bytes, 0, &mapped), "vkMapMemory");
        std::memcpy(mapped, rgba.data(), rgba.size());
        vkUnmapMemory(device_, rgba_staging_.memory);
        source = rgba_staging_.buffer;
    }

    wait_gpu();
    check(vkResetFences(device_, 1, &fence_), "vkResetFences");
    check(vkResetCommandBuffer(command_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command_, &begin), "vkBeginCommandBuffer");
    Frame& image = frames_[write_];
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = image.sampled ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL : VK_IMAGE_LAYOUT_UNDEFINED;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.srcAccessMask = image.sampled ? VK_ACCESS_SHADER_READ_BIT : 0;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.color;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(
        command_,
        image.sampled ? VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1, &barrier);
    if (device_rgba != VK_NULL_HANDLE) {
        // The source is the rasterizer's own buffer: make its compute writes
        // available to this command buffer's transfer read.
        VkBufferMemoryBarrier source_barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        source_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        source_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        source_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        source_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        source_barrier.buffer = device_rgba;
        source_barrier.offset = 0;
        source_barrier.size = device_rgba_bytes;
        vkCmdPipelineBarrier(
            command_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &source_barrier, 0, nullptr);
    }
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {width, height, 1};
    vkCmdCopyBufferToImage(
        command_, source, image.color, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        command_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    check(vkEndCommandBuffer(command_), "vkEndCommandBuffer");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_;
    check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");

    image.sampled = true;
    display_ = write_;
    write_ = (write_ + 1) % k_frames;
    cached_ = frame;
    cached_generation_ = generation_;
    cache_valid_ = true;
    target.slot = display_;
    target.view = image.view;
    target.sampler = image.sampler;
    target.width = width;
    target.height = height;
    return true;
}

bool Renderer::download_rgb(
    std::vector<std::uint8_t>& rgb, std::uint32_t& width, std::uint32_t& height,
    const std::uint32_t max_long_edge) {
    rgb.clear();
    width = 0;
    height = 0;
    if (display_ < 0 || !frames_[display_].color) return false;
    wait_gpu();
    const Frame& image = frames_[display_];
    std::uint32_t out_w = image.width;
    std::uint32_t out_h = image.height;
    if (max_long_edge > 0) {
        const std::uint32_t long_edge = std::max(out_w, out_h);
        if (long_edge > max_long_edge) {
            const double scale = static_cast<double>(max_long_edge) / long_edge;
            out_w = std::max(1U, static_cast<std::uint32_t>(std::lround(out_w * scale)));
            out_h = std::max(1U, static_cast<std::uint32_t>(std::lround(out_h * scale)));
        }
    }
    const VkDeviceSize bytes =
        static_cast<VkDeviceSize>(image.width) * image.height * 4U;
    Storage staging{};
    create_storage(
        staging, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    check(vkResetFences(device_, 1, &fence_), "vkResetFences");
    check(vkResetCommandBuffer(command_, 0), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command_, &begin), "vkBeginCommandBuffer");
    VkImageMemoryBarrier barrier{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    barrier.oldLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image.color;
    barrier.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(
        command_, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {image.width, image.height, 1};
    vkCmdCopyImageToBuffer(
        command_, image.color, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.buffer, 1,
        &region);
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(
        command_, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
        0, 0, nullptr, 0, nullptr, 1, &barrier);
    check(vkEndCommandBuffer(command_), "vkEndCommandBuffer");
    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &command_;
    check(vkQueueSubmit(queue_, 1, &submit, fence_), "vkQueueSubmit");
    wait_gpu();

    void* mapped{};
    check(vkMapMemory(device_, staging.memory, 0, bytes, 0, &mapped), "vkMapMemory");
    const auto* rgba = static_cast<const std::uint8_t*>(mapped);
    rgb.resize(static_cast<std::size_t>(out_w) * out_h * 3U);
    for (std::uint32_t y = 0; y < out_h; ++y) {
        const std::uint32_t sy = std::min(
            image.height - 1,
            static_cast<std::uint32_t>(
                (static_cast<double>(y) + 0.5) * image.height / out_h));
        for (std::uint32_t x = 0; x < out_w; ++x) {
            const std::uint32_t sx = std::min(
                image.width - 1,
                static_cast<std::uint32_t>(
                    (static_cast<double>(x) + 0.5) * image.width / out_w));
            const std::size_t src =
                (static_cast<std::size_t>(sy) * image.width + sx) * 4U;
            const std::size_t dst = (static_cast<std::size_t>(y) * out_w + x) * 3U;
            rgb[dst] = rgba[src];
            rgb[dst + 1] = rgba[src + 1];
            rgb[dst + 2] = rgba[src + 2];
        }
    }
    vkUnmapMemory(device_, staging.memory);
    destroy_storage(staging);
    width = out_w;
    height = out_h;
    return true;
}

}  // namespace splat_render
