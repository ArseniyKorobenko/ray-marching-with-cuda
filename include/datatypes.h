#pragma once
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector_types.h>

// #include <cmath>

namespace datatypes {
typedef float f32;
typedef double f64;

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;

typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef ptrdiff_t size;
typedef size_t usize;
typedef unsigned uint;

struct vec3;
struct mat3x3;
struct Color;

#define _HD_ __host__ __device__
#define _CHD_ constexpr __host__ __device__

// vec3(f32) constructor gives us operations with floats automatically.
// Might have to do that manually if it turns out slow.
// vec + vec; vec += vec, etc.
#define DEFINE_OPERATOR3(op)                             \
    _HD_ vec3 operator op(const vec3& rhs) const         \
    {                                                    \
        return vec3(x op rhs.x, y op rhs.y, z op rhs.z); \
    }                                                    \
    _HD_ vec3& operator op##=(const vec3& rhs)           \
    {                                                    \
        x op## = rhs.x;                                  \
        y op## = rhs.y;                                  \
        z op## = rhs.z;                                  \
        return *this;                                    \
    }
#define DEFINE_OPERATOR_MAT3(type, op)           \
    _HD_ type operator op(const type& rhs) const \
    {                                            \
        return {                                 \
            m[0] op rhs[0],                      \
            m[1] op rhs[1],                      \
            m[2] op rhs[2],                      \
        };                                       \
    }                                            \
    _HD_ type& operator op##=(const type& rhs)   \
    {                                            \
        m[0] op## = rhs[0];                      \
        m[1] op## = rhs[1];                      \
        m[2] op## = rhs[2];                      \
        return *this;                            \
    }

// Wrapper around cuda's float3 or float4.
struct vec3 : float4 {
    _HD_ vec3(const float3& vec)
    {
        x = vec.x;
        y = vec.y;
        z = vec.z;
    }
    _HD_ vec3(const vec3& vec)
    {
        x = vec.x;
        y = vec.y;
        z = vec.z;
    }
    _HD_ vec3(f32 e)
    {
        x = e;
        y = e;
        z = e;
    }
    _HD_ vec3(f32 e1, f32 e2, f32 e3)
    {
        x = e1;
        y = e2;
        z = e3;
    }

    DEFINE_OPERATOR3(+)
    DEFINE_OPERATOR3(-)
    DEFINE_OPERATOR3(*)
    DEFINE_OPERATOR3(/)
    _HD_ vec3 operator-() const { return vec3(-x, -y, -z); }
    _HD_ f32 operator[](int i) const { return reinterpret_cast<const f32*>(this)[i]; }

    _CHD_ f32 sum() const { return x + y + z; }
    _CHD_ f32 avg() const { return this->sum() / 3.f; }
    // Squared norm. Faster than norm.
    _CHD_ f32 norm2() const { return this->dot(*this); }
    // Aka length.
    _HD_ f32 norm() const { return sqrtf(this->norm2()); }
    // Call abs on all coordinates.
    _HD_ vec3 abs() const { return vec3(fabsf(x), fabsf(y), fabsf(z)); }
    // Set norm to 1, keeping direction.
    _HD_ vec3 normalized() const { return *this / this->norm(); }
    // (normalized() + 1) * 0.5
    _HD_ vec3 normalized0_1() const { return (this->normalized() + 1.f) * 0.5f; }

    _CHD_ f32 dot(const vec3& rhs) const { return x * rhs.x + y * rhs.y + z * rhs.z; }
    // Pairwise min.
    _HD_ vec3 min(const vec3& rhs) const
    {
        return vec3(fminf(x, rhs.x), fminf(y, rhs.y), fminf(z, rhs.z));
    }
    // Pairwise max.
    _HD_ vec3 max(const vec3& rhs) const
    {
        return vec3(fmaxf(x, rhs.x), fmaxf(y, rhs.y), fmaxf(z, rhs.z));
    }
    _HD_ vec3 cross(const vec3& rhs) const
    {
        f32 _x = y * rhs.z - z * rhs.y;
        f32 _y = z * rhs.x - x * rhs.z;
        f32 _z = x * rhs.y - y * rhs.x;
        return vec3(_x, _y, _z);
    }
    // Linear interpolation.
    static _HD_ vec3 lerp(const vec3& v0, const vec3& v1, f32 t)
    {
        return v0 * (1 - t) + v1 * t;
    }
};

inline std::ostream& operator<<(std::ostream& os, const vec3& vec)
{
    os << '(' << vec.x << ", " << vec.y << ", " << vec.z << ')';
    return os;
}
inline std::istream& operator>>(std::istream& is, vec3& vec)
{
    is >> vec.x >> vec.y >> vec.z;
    return is;
}

struct mat3x3 {
    vec3 m[3];
    _HD_ mat3x3()
        : m { 0, 0, 0 }
    {
    }
    _HD_ mat3x3(vec3 v1, vec3 v2, vec3 v3)
        : m { v1, v2, v3 }
    {
    }
    _HD_ vec3& operator[](int i) { return m[i]; }
    _HD_ const vec3& operator[](int i) const { return m[i]; }
    _HD_ vec3 operator*(const vec3& rhs) const
    {
        return vec3(
            m[0].dot(rhs),
            m[1].dot(rhs),
            m[2].dot(rhs));
    }
    _HD_ mat3x3 operator*(const mat3x3& rhs) const
    {
        mat3x3 t(rhs.T());
        return { t * m[0],
                 t * m[1],
                 t * m[2] };
    }
    DEFINE_OPERATOR_MAT3(mat3x3, +);
    DEFINE_OPERATOR_MAT3(mat3x3, -);
    _HD_ mat3x3 operator-() const { return { -m[0], -m[1], -m[2] }; }
    _HD_ mat3x3 T() const
    {
        return { { m[0][0], m[1][0], m[2][0] },
                 { m[0][1], m[1][1], m[2][1] },
                 { m[0][2], m[1][2], m[2][2] } };
    } // Rotation matrix from 3 orthonormal vectors.
    static _HD_ mat3x3 rotation(vec3 v1, vec3 v2, vec3 v3)
    {
        return mat3x3(v1, v2, v3);
    }
};

struct Color : vec3 {
    using vec3::vec3;
    _CHD_ f32 r() const { return x; }
    _CHD_ f32 g() const { return y; }
    _CHD_ f32 b() const { return z; }
    _CHD_ f32& r() { return x; }
    _CHD_ f32& g() { return y; }
    _CHD_ f32& b() { return z; }

    // r, g, b fields must be in [[0, 1]] range.
    _CHD_ u32 to_0BGR() const
    {
        u32 bgr = 0;
        bgr |= u32(this->b() * 255.99f) << 16;
        bgr |= u32(this->g() * 255.99f) << 8;
        bgr |= u32(this->r() * 255.99f);
        return bgr;
    }
    // r, g, b fields must be in [[0, 1]] range.
    _CHD_ u32 to_0RGB() const
    {
        u32 rgb = 0;
        rgb |= u32(this->r() * 255.99f) << 16;
        rgb |= u32(this->g() * 255.99f) << 8;
        rgb |= u32(this->b() * 255.99f);
        return rgb;
    }
    _CHD_ f32 luminance() const
    {
        return 0.2126f * this->r() + 0.7152f * this->g() + 0.0722f * this->b();
    }
};

inline _HD_ mat3x3 orthonormalize(vec3 e1, vec3 e2)
{
    vec3 new_e1 = e1.normalized();
    vec3 new_e3 = new_e1.cross(e2).normalized();

    vec3 new_e2 = new_e3.cross(new_e1);
    return mat3x3(new_e1, new_e2, new_e3);
}
inline _HD_ f32 norm2d(f32 x, f32 y) { return sqrtf(x * x + y * y); }
// Squared norm2d. Faster than norm2d.
inline _CHD_ f32 norm2d2(f32 x, f32 y) { return x * x + y * y; }
inline _HD_ f32 clip(f32 x, f32 xmin, f32 xmax) { return fminf(fmaxf(x, xmin), xmax); }

#undef _HD_
#undef _CHD_
}
