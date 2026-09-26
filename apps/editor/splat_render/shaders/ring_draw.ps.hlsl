// The quad edge is the splat_drender alpha = 1/255 contour. The interior is a
// disc with a flat outer rim so the outline stays readable on top of the splat.
struct PsIn {
    float4 position : SV_Position;
    nointerpolation float4 color : COLOR0;
    float2 uv : TEXCOORD0;
};

static const float k_exp4 = 0.01831563889;
static const float k_inv_exp4 = 1.01865736036;
static const float k_ring = 0.04;

float4 main(PsIn input) : SV_Target {
    float radius = dot(input.uv, input.uv);
    if (radius > 1.0) discard;
    float norm = (exp(radius * -4.0) - k_exp4) * k_inv_exp4;
    float alpha = max(0.05, norm * input.color.a);
    if (radius >= 1.0 - k_ring) alpha = 0.6;
    return float4(input.color.rgb, alpha);
}
