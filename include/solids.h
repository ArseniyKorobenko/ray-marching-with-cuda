#pragma once
#include "datatypes.h"
namespace solids_gpu {
using namespace datatypes;
#define SDF1(fn, ...) [=] __device__(vec3 v) { return fn(v, __VA_ARGS__); }

// Cube centered at 0. `dim` is its corner.
__device__ f32 cube(vec3 v, vec3 dim)
{
    vec3 d = v.abs() - dim;
    return fminf(fmaxf(d.x, fmaxf(d.y, d.z)), 0.f) + d.max(0.f).norm();
}
// Sphere centered at 0 with radius `r`.
__device__ f32 sphere(vec3 v, f32 r) { return v.norm() - r; }
// `n` is a unit normal vector. `r` is distance from 0.
__device__ f32 plane(vec3 v, vec3 n, f32 r) { return v.dot(n) - r; }
// y-aligned infinite cylinder centered at (`x`, `z`) with radius `r`.
__device__ f32 cylinder(vec3 v, f32 x, f32 z, f32 r)
{
    return norm2d(v.x - x, v.z - z) - r;
}
// x,z-aligned torus with radius `r1` and thickness `r2`
__device__ f32 torus(vec3 v, f32 r1, f32 r2)
{
    return norm2d((norm2d(v.x, v.z) - r1), v.y) - r2;
}

template <class F1, class F2>
__device__ f32 Union(vec3 v, F1 d1, F2 d2)
{
    return fminf(d1(v), d2(v));
};
template <class F1, class F2>
__device__ f32 Subtraction(vec3 v, F1 d1, F2 d2)
{
    return fmaxf(-d1(v), d2(v));
};
template <class F1, class F2>
__device__ f32 Intersection(vec3 v, F1 d1, F2 d2)
{
    return fmaxf(d1(v), d2(v));
};

template <class F1>
__device__ f32 Shift(vec3 v, vec3 sh, F1 d1) { return d1(v - sh); };
template <class F1>
__device__ f32 Rotate(vec3 v, mat3x3 t, F1 d1) { return d1(t * v); };
template <class F1>
__device__ f32 Scale(vec3 v, f32 s, F1 d1) { return d1(v / s) * s; };

__constant__ constexpr float3 lightPos { 10, 20, 3 };
constexpr int shadowIterations = 16;
constexpr int castIterations = 64;

template <class F>
__device__ vec3 localGrad(vec3 v, F Sdf)
{
    f32 d = Sdf(v);
    vec3 grad = vec3(Sdf(vec3(v.x + 0.0001f, v.y, v.z)) - d,
                     Sdf(vec3(v.x, v.y + 0.0001f, v.z)) - d,
                     Sdf(vec3(v.x, v.y, v.z + 0.0001f)) - d);
    return grad.normalized();
}

template <class F>
__device__ f32 shadow(vec3 v, vec3 lightDir, F Sdf)
{
    f32 kd = 1;
    int step = 0;
    for (f32 t = 0.1f; t < (vec3(lightPos) - v).norm()
                       && step < shadowIterations
                       && kd > 0.001f;) {
        f32 d = Sdf(v + lightDir * t);
        if (d < 0.001f) {
            kd = 0;
        } else {
            kd = fminf(kd, 16.f * d / t);
        }
        t += d;
        step++;
    }
    return kd;
}

template <class F>
__device__ f32 cast(vec3 ray, F Sdf)
{
    vec3 dir = ray.normalized();
    f32 dist = Sdf(ray);
    int i;
    for (i = 0; i < castIterations && dist > 0.001f; i++) {
        dist = Sdf(ray);
        ray += dir * dist;
    }
    if (dist > 0.001f)
        return 0;
    vec3 lightDir = (vec3(lightPos) - ray).normalized();
    vec3 n = localGrad(ray, Sdf);
    f32 light = clip((n.dot(lightDir) * 2.f + shadow(ray, lightDir, Sdf)) * 0.33f, 0.2f, 1.f);
    return light;
}
}

// Same thing but without __device__ tags.
namespace solids_cpu {
using namespace datatypes;
#define SDF2(fn, ...) [=](vec3 v) { return fn(v, __VA_ARGS__); }

// Cube centered at 0. `dim` is its corner.
f32 cube(vec3 v, vec3 dim)
{
    vec3 d = v.abs() - dim;
    return fminf(fmaxf(d.x, fmaxf(d.y, d.z)), 0.f) + d.max(0.f).norm();
}
// Sphere centered at 0 with radius `r`.
f32 sphere(vec3 v, f32 r) { return v.norm() - r; }
// `n` is a unit normal vector. `r` is distance from 0.
f32 plane(vec3 v, vec3 n, f32 r) { return v.dot(n) - r; }
// y-aligned infinite cylinder centered at (`x`, `z`) with radius `r`.
f32 cylinder(vec3 v, f32 x, f32 z, f32 r)
{
    return norm2d(v.x - x, v.z - z) - r;
}
// x,z-aligned torus with radius `r1` and thickness `r2`
f32 torus(vec3 v, f32 r1, f32 r2)
{
    return norm2d((norm2d(v.x, v.z) - r1), v.y) - r2;
}

template <class F1, class F2>
f32 Union(vec3 v, F1 d1, F2 d2)
{
    return fminf(d1(v), d2(v));
};
template <class F1, class F2>
f32 Subtraction(vec3 v, F1 d1, F2 d2)
{
    return fmaxf(-d1(v), d2(v));
};
template <class F1, class F2>
f32 Intersection(vec3 v, F1 d1, F2 d2)
{
    return fmaxf(d1(v), d2(v));
};

template <class F1>
f32 Shift(vec3 v, vec3 sh, F1 d1) { return d1(v - sh); };
template <class F1>
f32 Rotate(vec3 v, mat3x3 t, F1 d1) { return d1(t * v); };
template <class F1>
f32 Scale(vec3 v, f32 s, F1 d1) { return d1(v / s) * s; };

__constant__ constexpr float3 lightPos { 10, 20, 3 };
constexpr int shadowIterations = 16;
constexpr int castIterations = 64;

template <class F>
vec3 localGrad(vec3 v, F Sdf)
{
    f32 d = Sdf(v);
    vec3 grad = vec3(Sdf(vec3(v.x + 0.0001f, v.y, v.z)) - d,
                     Sdf(vec3(v.x, v.y + 0.0001f, v.z)) - d,
                     Sdf(vec3(v.x, v.y, v.z + 0.0001f)) - d);
    return grad.normalized();
}

template <class F>
f32 shadow(vec3 v, vec3 lightDir, F Sdf)
{
    f32 kd = 1;
    int step = 0;
    for (f32 t = 0.1f; t < (vec3(lightPos) - v).norm()
                       && step < shadowIterations
                       && kd > 0.001f;) {
        f32 d = Sdf(v + lightDir * t);
        if (d < 0.001f) {
            kd = 0;
        } else {
            kd = fminf(kd, 16.f * d / t);
        }
        t += d;
        step++;
    }
    return kd;
}

template <class F>
f32 cast(vec3 ray, F Sdf)
{
    vec3 dir = ray.normalized();
    f32 dist = Sdf(ray);
    int i;
    for (i = 0; i < castIterations && dist > 0.001f; i++) {
        dist = Sdf(ray);
        ray += dir * dist;
    }
    if (dist > 0.001f)
        return 0;
    vec3 lightDir = (vec3(lightPos) - ray).normalized();
    vec3 n = localGrad(ray, Sdf);
    f32 light = clip((n.dot(lightDir) * 2.f + shadow(ray, lightDir, Sdf)) * 0.33f, 0.2f, 1.f);
    return light;
}
}
