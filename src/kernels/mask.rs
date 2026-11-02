//! Lineage bitmasks for the coalescent DPs.
//!
//! A gene tree with ``n`` nodes needs ``n`` bits.  Small trees use a
//! machine word; larger ones use fixed-width multiword masks, chosen once
//! per gene tree by [`dispatch`](crate::kernels::msnc) on ``n``.

use std::fmt::Debug;
use std::hash::Hash;

use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyInt};

pub trait Mask: Copy + Eq + Hash + Default + Debug + Send + Sync + 'static {
    const BITS: usize;
    fn zero() -> Self;
    fn bit(i: usize) -> Self;
    fn is_zero(&self) -> bool;
    fn popcount(&self) -> u32;
    fn and(self, o: Self) -> Self;
    fn or(self, o: Self) -> Self;
    fn xor(self, o: Self) -> Self;
    /// ``self & !o``
    fn andnot(self, o: Self) -> Self;
    fn test(&self, i: usize) -> bool;
    /// Index of the lowest set bit (``self`` must be non-zero).
    fn lowest(&self) -> usize;
    /// ``self`` with its lowest set bit cleared.
    fn clear_lowest(self) -> Self;
    /// ``(self - 1) & universe``: the next smaller subset of ``universe``.
    fn pred_and(self, universe: Self) -> Self;
    fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self>;
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>;

    #[inline]
    fn overlaps(&self, o: &Self) -> bool {
        !self.and(*o).is_zero()
    }

    /// Set bit indices, low to high.
    fn bits(&self) -> Vec<usize> {
        let mut out = Vec::with_capacity(self.popcount() as usize);
        let mut m = *self;
        while !m.is_zero() {
            out.push(m.lowest());
            m = m.clear_lowest();
        }
        out
    }
}

impl Mask for u64 {
    const BITS: usize = 64;
    #[inline]
    fn zero() -> Self {
        0
    }
    #[inline]
    fn bit(i: usize) -> Self {
        1u64 << i
    }
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0
    }
    #[inline]
    fn popcount(&self) -> u32 {
        self.count_ones()
    }
    #[inline]
    fn and(self, o: Self) -> Self {
        self & o
    }
    #[inline]
    fn or(self, o: Self) -> Self {
        self | o
    }
    #[inline]
    fn xor(self, o: Self) -> Self {
        self ^ o
    }
    #[inline]
    fn andnot(self, o: Self) -> Self {
        self & !o
    }
    #[inline]
    fn test(&self, i: usize) -> bool {
        (self >> i) & 1 == 1
    }
    #[inline]
    fn lowest(&self) -> usize {
        self.trailing_zeros() as usize
    }
    #[inline]
    fn clear_lowest(self) -> Self {
        self & (self - 1)
    }
    #[inline]
    fn pred_and(self, universe: Self) -> Self {
        self.wrapping_sub(1) & universe
    }
    fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        obj.extract::<u64>()
    }
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.into_pyobject(py)?.into_any())
    }
}

impl Mask for u128 {
    const BITS: usize = 128;
    #[inline]
    fn zero() -> Self {
        0
    }
    #[inline]
    fn bit(i: usize) -> Self {
        1u128 << i
    }
    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0
    }
    #[inline]
    fn popcount(&self) -> u32 {
        self.count_ones()
    }
    #[inline]
    fn and(self, o: Self) -> Self {
        self & o
    }
    #[inline]
    fn or(self, o: Self) -> Self {
        self | o
    }
    #[inline]
    fn xor(self, o: Self) -> Self {
        self ^ o
    }
    #[inline]
    fn andnot(self, o: Self) -> Self {
        self & !o
    }
    #[inline]
    fn test(&self, i: usize) -> bool {
        (self >> i) & 1 == 1
    }
    #[inline]
    fn lowest(&self) -> usize {
        self.trailing_zeros() as usize
    }
    #[inline]
    fn clear_lowest(self) -> Self {
        self & (self - 1)
    }
    #[inline]
    fn pred_and(self, universe: Self) -> Self {
        self.wrapping_sub(1) & universe
    }
    fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        obj.extract::<u128>()
    }
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.into_pyobject(py)?.into_any())
    }
}

/// A fixed-width mask of ``N`` 64-bit words.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Words<const N: usize>(pub [u64; N]);

impl<const N: usize> Default for Words<N> {
    fn default() -> Self {
        Words([0; N])
    }
}

impl<const N: usize> Mask for Words<N> {
    const BITS: usize = 64 * N;
    #[inline]
    fn zero() -> Self {
        Words([0; N])
    }
    #[inline]
    fn bit(i: usize) -> Self {
        let mut w = [0u64; N];
        w[i / 64] = 1u64 << (i % 64);
        Words(w)
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.0.iter().all(|&w| w == 0)
    }
    #[inline]
    fn popcount(&self) -> u32 {
        self.0.iter().map(|w| w.count_ones()).sum()
    }
    #[inline]
    fn and(self, o: Self) -> Self {
        let mut w = [0u64; N];
        for i in 0..N {
            w[i] = self.0[i] & o.0[i];
        }
        Words(w)
    }
    #[inline]
    fn or(self, o: Self) -> Self {
        let mut w = [0u64; N];
        for i in 0..N {
            w[i] = self.0[i] | o.0[i];
        }
        Words(w)
    }
    #[inline]
    fn xor(self, o: Self) -> Self {
        let mut w = [0u64; N];
        for i in 0..N {
            w[i] = self.0[i] ^ o.0[i];
        }
        Words(w)
    }
    #[inline]
    fn andnot(self, o: Self) -> Self {
        let mut w = [0u64; N];
        for i in 0..N {
            w[i] = self.0[i] & !o.0[i];
        }
        Words(w)
    }
    #[inline]
    fn test(&self, i: usize) -> bool {
        (self.0[i / 64] >> (i % 64)) & 1 == 1
    }
    #[inline]
    fn lowest(&self) -> usize {
        for (k, &w) in self.0.iter().enumerate() {
            if w != 0 {
                return k * 64 + w.trailing_zeros() as usize;
            }
        }
        usize::MAX
    }
    #[inline]
    fn clear_lowest(self) -> Self {
        let mut w = self.0;
        for item in w.iter_mut() {
            if *item != 0 {
                *item &= *item - 1;
                break;
            }
        }
        Words(w)
    }
    #[inline]
    fn pred_and(self, universe: Self) -> Self {
        // Multiword subtraction of 1 with borrow, then mask.
        let mut w = self.0;
        for item in w.iter_mut() {
            if *item == 0 {
                *item = u64::MAX;
            } else {
                *item -= 1;
                break;
            }
        }
        for i in 0..N {
            w[i] &= universe.0[i];
        }
        Words(w)
    }
    fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        let py = obj.py();
        let int = obj.cast::<PyInt>()?;
        let bytes = int
            .call_method1("to_bytes", (N * 8, "little"))?
            .cast_into::<PyBytes>()?;
        let b = bytes.as_bytes();
        let mut w = [0u64; N];
        for i in 0..N {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(&b[i * 8..i * 8 + 8]);
            w[i] = u64::from_le_bytes(arr);
        }
        let _ = py;
        Ok(Words(w))
    }
    fn to_py<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let mut b = Vec::with_capacity(N * 8);
        for w in self.0.iter() {
            b.extend_from_slice(&w.to_le_bytes());
        }
        let bytes = PyBytes::new(py, &b);
        py.get_type::<PyInt>()
            .call_method1("from_bytes", (bytes, "little"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_pred_and_enumerates_subsets() {
        let universe = Words::<2>::bit(0).or(Words::<2>::bit(70));
        let mut s = universe;
        let mut seen = 0;
        loop {
            seen += 1;
            if s.is_zero() {
                break;
            }
            s = s.pred_and(universe);
        }
        assert_eq!(seen, 4);
    }

    #[test]
    fn words_bits_round_trip() {
        let m = Words::<4>::bit(3).or(Words::<4>::bit(200));
        assert_eq!(m.bits(), vec![3, 200]);
        assert_eq!(m.popcount(), 2);
        assert!(m.test(200) && !m.test(199));
        assert_eq!(m.clear_lowest().bits(), vec![200]);
    }
}
