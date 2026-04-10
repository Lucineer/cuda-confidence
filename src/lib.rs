/*!
# cuda-confidence

The confidence primitive — every value carries its own uncertainty.

This is the keystone of the entire Lucineer architecture.

In normal computing, a number is just a number. 42 is 42.
In confidence computing, 42 is "42, and I'm 73% sure about that."

Every value. Every computation. Every message between agents.
Confidence propagates through addition, multiplication, comparison.
It fuses when agents share data. It decays with time. It grows with evidence.

This crate provides the foundational Conf type and all operations on it.
Every other crate in the fleet builds on this.
*/

use serde::{Deserialize, Serialize};
use std::ops::{Add, Sub, Mul, Div};
use std::cmp::Ordering;

/// A value with confidence
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Conf {
    pub value: f64,
    pub confidence: f64, // [0, 1]
}

impl Conf {
    pub fn new(value: f64, confidence: f64) -> Self {
        Conf { value, confidence: confidence.clamp(0.0, 1.0) }
    }

    pub fn certain(value: f64) -> Self { Conf { value, confidence: 1.0 } }
    pub fn unknown() -> Self { Conf { value: 0.0, confidence: 0.0 } }
    pub fn half(value: f64) -> Self { Conf { value, confidence: 0.5 } }

    /// Is this value meaningful? (confidence above threshold)
    pub fn is_meaningful(&self, threshold: f64) -> bool { self.confidence >= threshold }

    /// Absolute uncertainty
    pub fn uncertainty(&self) -> f64 { 1.0 - self.confidence }

    /// Bayesian fusion: combine two independent estimates of the same value
    /// p = 1 / (1/c1 + 1/c2), v = weighted average
    pub fn fuse(&self, other: &Conf) -> Conf {
        if self.confidence < 0.001 && other.confidence < 0.001 { return Conf::unknown(); }

        let total_conf = self.confidence + other.confidence;
        if total_conf < 0.001 { return Conf::unknown(); }

        let fused_conf = 1.0 / (1.0 / self.confidence.max(0.001) + 1.0 / other.confidence.max(0.001));
        let fused_value = (self.value * self.confidence + other.value * other.confidence) / total_conf;

        Conf { value: fused_value, confidence: fused_conf.clamp(0.0, 1.0) }
    }

    /// Decay confidence by a factor (time passing, distance, etc)
    pub fn decay(&self, factor: f64) -> Conf {
        Conf { value: self.value, confidence: self.confidence * factor.clamp(0.0, 1.0) }
    }

    /// Boost confidence with evidence
    pub fn boost(&self, evidence: f64) -> Conf {
        let new_conf = (self.confidence + evidence * (1.0 - self.confidence)).clamp(0.0, 1.0);
        Conf { value: self.value, confidence: new_conf }
    }

    /// Apply a function to the value, reducing confidence slightly
    pub fn map(&self, f: impl Fn(f64) -> f64, confidence_cost: f64) -> Conf {
        Conf { value: f(self.value), confidence: (self.confidence * (1.0 - confidence_cost)).clamp(0.0, 1.0) }
    }

    /// With minimum confidence floor
    pub fn min_conf(&self, floor: f64) -> Conf {
        Conf { value: self.value, confidence: self.confidence.max(floor) }
    }

    /// To optional — None if below threshold
    pub fn to_option(&self, threshold: f64) -> Option<f64> {
        if self.is_meaningful(threshold) { Some(self.value) } else { None }
    }
}

// Arithmetic: confidence = geometric mean of operands
impl Add for Conf {
    type Output = Conf;
    fn add(self, rhs: Conf) -> Conf {
        let conf = (self.confidence * rhs.confidence).sqrt();
        Conf { value: self.value + rhs.value, confidence: conf }
    }
}

impl Sub for Conf {
    type Output = Conf;
    fn sub(self, rhs: Conf) -> Conf {
        let conf = (self.confidence * rhs.confidence).sqrt();
        Conf { value: self.value - rhs.value, confidence: conf }
    }
}

impl Mul for Conf {
    type Output = Conf;
    fn mul(self, rhs: Conf) -> Conf {
        let conf = (self.confidence * rhs.confidence).sqrt();
        Conf { value: self.value * rhs.value, confidence: conf }
    }
}

impl Div for Conf {
    type Output = Conf;
    fn div(self, rhs: Conf) -> Conf {
        if rhs.value.abs() < 1e-10 { return Conf::unknown(); }
        let conf = (self.confidence * rhs.confidence).sqrt();
        Conf { value: self.value / rhs.value, confidence: conf }
    }
}

// Comparison with confidence penalty
impl PartialEq for Conf {
    fn eq(&self, other: &Self) -> bool {
        let threshold = (self.confidence * other.confidence).sqrt();
        threshold > 0.5 && (self.value - other.value).abs() < 0.001
    }
}

impl PartialOrd for Conf {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let threshold = (self.confidence * other.confidence).sqrt();
        if threshold < 0.3 { return None; } // too uncertain to compare
        let penalty = 1.0 - threshold;
        let effective_diff = (self.value - other.value).abs();
        let tolerance = penalty * self.value.abs().max(1.0) * 0.1;
        if effective_diff < tolerance { Some(Ordering::Equal) }
        else { self.value.partial_cmp(&other.value) }
    }
}

/// Confidence distribution — track multiple confidence values
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConfidenceDist {
    pub values: Vec<Conf>,
}

impl ConfidenceDist {
    pub fn new() -> Self { ConfidenceDist { values: vec![] } }
    pub fn push(&mut self, conf: Conf) { self.values.push(conf); }

    /// Fuse all values into one
    pub fn fuse_all(&self) -> Conf {
        if self.values.is_empty() { return Conf::unknown(); }
        self.values.iter().skip(1).fold(self.values[0], |acc, v| acc.fuse(v))
    }

    /// Average confidence
    pub fn avg_confidence(&self) -> f64 {
        if self.values.is_empty() { return 0.0; }
        self.values.iter().map(|v| v.confidence).sum::<f64>() / self.values.len() as f64
    }

    /// Count values above threshold
    pub fn meaningful_count(&self, threshold: f64) -> usize {
        self.values.iter().filter(|v| v.is_meaningful(threshold)).count()
    }
}

/// Confidence gate — only pass value if confidence exceeds threshold
pub fn conf_gate(input: &Conf, threshold: f64, default: f64) -> f64 {
    if input.is_meaningful(threshold) { input.value } else { default }
}

/// Weighted confidence merge for agent consensus
pub fn consensus(values: &[Conf], min_agreement: f64) -> Conf {
    if values.is_empty() { return Conf::unknown(); }

    let fused = values.iter().skip(1).fold(values[0], |acc, v| acc.fuse(v));

    // Check agreement: how close are all values?
    let avg = values.iter().map(|v| v.value).sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v.value - avg).powi(2)).sum::<f64>() / values.len() as f64;
    let agreement = 1.0 / (1.0 + variance.sqrt());

    Conf { value: fused.value, confidence: fused.confidence * agreement }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_certain() {
        let c = Conf::certain(42.0);
        assert_eq!(c.value, 42.0);
        assert_eq!(c.confidence, 1.0);
    }

    #[test]
    fn test_unknown() {
        let u = Conf::unknown();
        assert_eq!(u.confidence, 0.0);
        assert!(!u.is_meaningful(0.5));
    }

    #[test]
    fn test_confidence_clamp() {
        let c = Conf::new(1.0, 1.5);
        assert_eq!(c.confidence, 1.0);
        let c = Conf::new(1.0, -0.5);
        assert_eq!(c.confidence, 0.0);
    }

    #[test]
    fn test_addition() {
        let a = Conf::certain(3.0);
        let b = Conf::certain(4.0);
        let sum = a + b;
        assert_eq!(sum.value, 7.0);
        assert_eq!(sum.confidence, 1.0); // sqrt(1*1)
    }

    #[test]
    fn test_multiplication() {
        let a = Conf::new(3.0, 0.8);
        let b = Conf::new(4.0, 0.9);
        let prod = a * b;
        assert_eq!(prod.value, 12.0);
        // conf = sqrt(0.8*0.9) ≈ 0.849
        assert!((prod.confidence - 0.849).abs() < 0.01);
    }

    #[test]
    fn test_fusion() {
        let a = Conf::new(10.0, 0.8);
        let b = Conf::new(12.0, 0.6);
        let fused = a.fuse(&b);
        // Bayesian: conf = 1/(1/0.8 + 1/0.6) = 1/(1.25+1.667) = 1/2.917 ≈ 0.343
        // value weighted: (10*0.8 + 12*0.6)/(0.8+0.6) = 15.2/1.4 ≈ 10.857
        assert!((fused.confidence - 0.343).abs() < 0.01);
    }

    #[test]
    fn test_decay() {
        let c = Conf::certain(5.0);
        let decayed = c.decay(0.5);
        assert_eq!(decayed.value, 5.0);
        assert_eq!(decayed.confidence, 0.5);
    }

    #[test]
    fn test_boost() {
        let c = Conf::half(5.0);
        let boosted = c.boost(0.5);
        // 0.5 + 0.5*(1-0.5) = 0.75
        assert!((boosted.confidence - 0.75).abs() < 0.01);
    }

    #[test]
    fn test_gate() {
        let pass = Conf::new(42.0, 0.9);
        let fail = Conf::new(42.0, 0.3);
        assert_eq!(conf_gate(&pass, 0.5, -1.0), 42.0);
        assert_eq!(conf_gate(&fail, 0.5, -1.0), -1.0);
    }

    #[test]
    fn test_map() {
        let c = Conf::certain(5.0);
        let doubled = c.map(|v| v * 2.0, 0.05);
        assert_eq!(doubled.value, 10.0);
        assert!(doubled.confidence < 1.0); // slight cost
    }

    #[test]
    fn test_comparison() {
        let a = Conf::certain(5.0);
        let b = Conf::certain(3.0);
        assert!(a > b);
    }

    #[test]
    fn test_uncertain_comparison() {
        let a = Conf::new(5.0, 0.1);
        let b = Conf::new(3.0, 0.1);
        assert!(a.partial_cmp(&b).is_none()); // too uncertain
    }

    #[test]
    fn test_distribution_fuse() {
        let mut dist = ConfidenceDist::new();
        dist.push(Conf::new(10.0, 0.8));
        dist.push(Conf::new(12.0, 0.6));
        let fused = dist.fuse_all();
        assert!(fused.confidence > 0.3);
    }

    #[test]
    fn test_consensus() {
        let values = vec![Conf::new(10.0, 0.9), Conf::new(10.1, 0.9), Conf::new(9.9, 0.9)];
        let c = consensus(&values, 0.5);
        assert!((c.value - 10.0).abs() < 0.2);
        assert!(c.confidence > 0.8); // high agreement
    }

    #[test]
    fn test_consensus_disagreement() {
        let values = vec![Conf::new(1.0, 0.9), Conf::new(100.0, 0.9)];
        let c = consensus(&values, 0.5);
        assert!(c.confidence < 0.5); // low agreement
    }

    #[test]
    fn test_to_option() {
        let good = Conf::new(5.0, 0.8);
        let bad = Conf::new(5.0, 0.2);
        assert_eq!(good.to_option(0.5), Some(5.0));
        assert_eq!(bad.to_option(0.5), None);
    }

    #[test]
    fn test_division_by_zero() {
        let a = Conf::certain(5.0);
        let b = Conf::certain(0.0);
        let result = a / b;
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_min_conf() {
        let c = Conf::new(5.0, 0.2);
        let floored = c.min_conf(0.5);
        assert_eq!(floored.confidence, 0.5);
    }
}
