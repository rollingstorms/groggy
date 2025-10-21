//! Math utilities for interaction handling
#![allow(clippy::wrong_self_convention)]

#[inline]
pub fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

#[derive(Clone, Copy, Debug)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Vec3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    // Note: Intentionally not implementing Add/Sub/Mul traits to avoid operator overloading
    #[allow(clippy::should_implement_trait)]
    pub fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }

    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, scalar: f64) -> Self {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }

    pub fn dot(self, other: Self) -> f64 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }

    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }

    pub fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn normalized(self) -> Self {
        let n = self.norm();
        if n > 0.0 {
            self.mul(1.0 / n)
        } else {
            self
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Quat {
    pub w: f64,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Quat {
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    pub fn from_axis_angle(axis: Vec3, angle: f64) -> Self {
        let axis = axis.normalized();
        let half = angle * 0.5;
        let sin_half = half.sin();
        Self {
            w: half.cos(),
            x: axis.x * sin_half,
            y: axis.y * sin_half,
            z: axis.z * sin_half,
        }
        .normalized()
    }

    pub fn normalized(mut self) -> Self {
        let n = (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt();
        if n > 0.0 {
            self.w /= n;
            self.x /= n;
            self.y /= n;
            self.z /= n;
        }
        self
    }

    // Note: Intentionally not implementing Mul trait to avoid operator overloading
    #[allow(clippy::should_implement_trait)]
    pub fn mul(self, other: Self) -> Self {
        Self {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
        .normalized()
    }

    pub fn rotate_vec3(self, v: Vec3) -> Vec3 {
        let qv = Quat {
            w: 0.0,
            x: v.x,
            y: v.y,
            z: v.z,
        };
        let qi = Quat {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        };
        let r = self.mul(qv).mul(qi);
        Vec3::new(r.x, r.y, r.z)
    }
}
