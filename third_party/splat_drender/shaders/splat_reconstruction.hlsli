// Camera pack: placement records, then enable + 8 scalar animation values,
// followed by the model-sized arrival-time array. Training leaves it disabled.
uint reconstruction_base() { return 23u + 9u * asuint(camera[22]); }
bool reconstruction_enabled() { return camera[reconstruction_base()] > .5f; }
uint reconstruction_hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15;
    x *= 0x846ca68bU; return x ^ (x >> 16);
}
float reconstruction_smooth(float a,float b,float x) {
    float u=saturate((x-a)/(b-a)); return u*u*(3.f-2.f*u);
}
bool reconstruction_complete(uint index) {
    uint b=reconstruction_base();
    return camera[b+1]>=camera[b+3] || camera[b+1]-camera[b+9+index]>=camera[b+4]+camera[b+5];
}
void reconstruction_geometry(uint index,inout float3 scale,inout float opacity) {
    if (!reconstruction_enabled()) return;
    if (reconstruction_complete(index)) return;
    uint b=reconstruction_base();
    float time=camera[b+1],age=time-camera[b+9+index];
    float hold=camera[b+4],growth=camera[b+5];
    float grow=reconstruction_smooth(hold,hold+growth*.88f,age);
    float seed=float(reconstruction_hash(index)%100000u)/100000.f;
    float illuminate=reconstruction_smooth(-.75f,-.12f,age);
    float alpha=age<-.75f
        ? (seed<.008f ? .4f*reconstruction_smooth(camera[b+2]-.7f,camera[b+2]+.3f,time) : .00001f)
        : max(opacity*grow,(seed<illuminate*camera[b+7] ? .72f : .00001f)*(1.f-grow));
    opacity=clamp(alpha,.000001f,.999999f);
    float radius=camera[b+6]*(age<-.75f ? .00026f/.00036f : 1.f);
    scale=exp(lerp(log(radius).xxx,log(max(scale,1.e-20f)),grow));
}
float3 reconstruction_color(uint index,float3 original) {
    if (!reconstruction_enabled()) return original;
    if (reconstruction_complete(index)) return max(original,0.f);
    uint b=reconstruction_base();
    float age=camera[b+1]-camera[b+9+index],hold=camera[b+4],growth=camera[b+5];
    float material=reconstruction_smooth(hold+growth*.04f,hold+growth,age);
    float glow=(1.f-material)*reconstruction_smooth(-.75f,-.12f,age)*camera[b+8]/.7f;
    return max(float3(.025f,.48f,1.05f)*(1.f-material)+original*material+glow*float3(.01f,.16f,.16f),0.f);
}
