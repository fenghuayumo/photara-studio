#include "splat_math.hlsli"
StructuredBuffer<float> prediction : register(t0);
StructuredBuffer<float> target : register(t1);
StructuredBuffer<float> mask_image : register(t2);
RWStructuredBuffer<float> work0 : register(u3);
RWStructuredBuffer<float> work1 : register(u4);
RWStructuredBuffer<float> work2 : register(u5);
RWStructuredBuffer<float> work3 : register(u6);
RWStructuredBuffer<float> work4 : register(u7);
RWStructuredBuffer<float> derivative0 : register(u8);
RWStructuredBuffer<float> derivative1 : register(u9);
RWStructuredBuffer<float> derivative2 : register(u10);
RWStructuredBuffer<float> output : register(u11);

static const float gaussian[11] = {
    0.001028380123898387f, 0.0075987582094967365f,
    0.036000773310661316f, 0.10936068743467331f,
    0.21300552785396576f, 0.26601171493530273f,
    0.21300552785396576f, 0.10936068743467331f,
    0.036000773310661316f, 0.0075987582094967365f,
    0.001028380123898387f};
float push_float(uint value) { return asfloat(value); }

float prediction_at(int c, int x, int y, uint w, uint h, bool masked) {
    if (x < 0 || y < 0 || x >= (int)w || y >= (int)h) return 0.0f;
    uint p = (uint)y * w + (uint)x;
    float v = prediction[(uint)c * w * h + p];
    return masked ? v * mask_image[p] : v;
}
float target_at(int c, int x, int y, uint w, uint h, bool masked) {
    if (x < 0 || y < 0 || x >= (int)w || y >= (int)h) return 0.0f;
    uint p = (uint)y * w + (uint)x;
    float v = target[(uint)c * w * h + p];
    return masked ? v * mask_image[p] : v;
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint width = pc.u0, height = pc.u1;
    uint x = id.x, y = id.y, channel = id.z;
    if (x >= width || y >= height || channel >= 3u) return;
    uint pixels = width * height;
    uint index = channel * pixels + y * width + x;
    bool masked = pc.u3 != 0u;
    float ssim_weight = push_float(pc.u2);
    float normalization = push_float(pc.u4);
    uint pass = pc.u5;
    if (pass == 0u) {
        float s0=0, s1=0, s2=0, s3=0, s4=0;
        [loop] for (int d=-5; d<=5; ++d) {
            float a=prediction_at((int)channel,(int)x+d,(int)y,width,height,masked);
            float b=target_at((int)channel,(int)x+d,(int)y,width,height,masked);
            float q=gaussian[d+5];
            s0+=q*a; s1+=q*a*a; s2+=q*b; s3+=q*b*b; s4+=q*a*b;
        }
        work0[index]=s0; work1[index]=s1; work2[index]=s2;
        work3[index]=s3; work4[index]=s4; return;
    }
    if (pass == 1u) {
        float mu1=0, moment1=0, mu2=0, moment2=0, cross=0;
        [loop] for (int d=-5; d<=5; ++d) {
            int sy=(int)y+d;
            if (sy<0 || sy>=(int)height) continue;
            uint source=channel*pixels+(uint)sy*width+x;
            float q=gaussian[d+5];
            mu1+=q*work0[source]; moment1+=q*work1[source];
            mu2+=q*work2[source]; moment2+=q*work3[source]; cross+=q*work4[source];
        }
        float variance1=moment1-mu1*mu1, variance2=moment2-mu2*mu2;
        float covariance=cross-mu1*mu2;
        float a=mu1*mu1+mu2*mu2+0.0001f, b=variance1+variance2+0.0009f;
        float c=2.0f*mu1*mu2+0.0001f, d=2.0f*covariance+0.0009f;
        float ssim=c*d/(a*b);
        derivative0[index]=-ssim_weight*normalization*((2.0f*mu2*d)/(a*b)-(2.0f*mu2*c)/(a*b)-(2.0f*mu1*c*d)/(a*a*b)+(2.0f*mu1*c*d)/(a*b*b));
        derivative1[index]=-ssim_weight*normalization*(-c*d/(a*b*b));
        derivative2[index]=-ssim_weight*normalization*(2.0f*c/(a*b));
        float center1=prediction_at((int)channel,(int)x,(int)y,width,height,masked);
        float center2=target_at((int)channel,(int)x,(int)y,width,height,masked);
        output[3u*pixels+index]=ssim_weight*(1.0f-ssim)+(1.0f-ssim_weight)*abs(center1-center2);
        return;
    }
    if (pass == 2u) {
        float s0=0, s1=0, s2=0;
        [loop] for (int d=-5; d<=5; ++d) {
            int ox=(int)x+d;
            if (ox<5 || ox>=(int)width-5 || y<5u || y>=height-5u) continue;
            uint source=channel*pixels+y*width+(uint)ox;
            float q=gaussian[d+5];
            s0+=q*derivative0[source]; s1+=q*derivative1[source]; s2+=q*derivative2[source];
        }
        work0[index]=s0; work1[index]=s1; work2[index]=s2; return;
    }
    float s0=0, s1=0, s2=0;
    [loop] for (int d=-5; d<=5; ++d) {
        int oy=(int)y+d;
        if (oy<0 || oy>=(int)height) continue;
        uint source=channel*pixels+(uint)oy*width+x;
        float q=gaussian[d+5];
        s0+=q*work0[source]; s1+=q*work1[source]; s2+=q*work2[source];
    }
    float pixel1=prediction_at((int)channel,(int)x,(int)y,width,height,masked);
    float pixel2=target_at((int)channel,(int)x,(int)y,width,height,masked);
    float sign_value=pixel1>pixel2?1.0f:pixel1<pixel2?-1.0f:0.0f;
    float l1_chain=x>=5u&&x<width-5u&&y>=5u&&y<height-5u?normalization:0.0f;
    float valid=masked?mask_image[y*width+x]:1.0f;
    output[index]=(s0+2.0f*pixel1*s1+pixel2*s2+(1.0f-ssim_weight)*sign_value*l1_chain)*valid;
}
