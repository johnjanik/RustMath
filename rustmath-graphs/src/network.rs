//! Capacitated networks and network flow (MAGMA Handbook Chapter 151
//! "Networks").
//!
//! A MAGMA `GrphNet` is a directed multigraph whose arcs carry a non-negative
//! integer capacity (and optionally a cost).  This module provides the
//! [`Network`] type and the flow machinery of §151.4:
//!
//!   * `MaximumFlow(s, t)` -> [`Network::maximum_flow`] (Dinic's algorithm)
//!   * `MinimumCut(s, t)`  -> [`Network::minimum_cut`] (min-cut from the residual)
//!   * `Flow(e)`           -> [`Network::flow_on_arc`]
//!   * minimum-cost maximum flow (the fundamental problem of §151.1) ->
//!     [`Network::min_cost_max_flow`] (successive shortest paths / SPFA)
//!
//! Capacities and costs are `rustmath-integers::Integer` (bignum), exactly as
//! the port plan requires ("network flow is defined over integer capacities so
//! use rustmath-integers Integer to avoid overflow on large capacities").  The
//! `Network` container itself is a plain combinatorial object; the algebraic
//! objects it consumes/produces (capacities, flow values, costs) are `Integer`.
//!
//! Arcs are stored in adjacent forward/reverse pairs so the residual of arc `a`
//! is arc `a ^ 1`.  All flow algorithms operate on this residual representation.

use rustmath_integers::Integer;
use std::collections::VecDeque;

/// A capacitated (and optionally costed) directed network over `Integer`.
#[derive(Debug, Clone)]
pub struct Network {
    n: usize,
    /// Source vertex of each arc.
    tail: Vec<usize>,
    /// Target vertex of each arc.
    head: Vec<usize>,
    /// Residual capacity of each arc (mutated by the flow algorithms).
    cap: Vec<Integer>,
    /// Original capacity of each arc (forward arcs keep their capacity; the
    /// paired reverse arc has original capacity 0).
    orig_cap: Vec<Integer>,
    /// Per-unit cost of each arc (reverse arc = negated cost).
    cost: Vec<Integer>,
    /// Adjacency: vertex -> list of arc indices leaving it.
    adj: Vec<Vec<usize>>,
    // transient state for Dinic
    level: Vec<i64>,
    iter: Vec<usize>,
}

impl Network {
    /// Create an empty network on `n` vertices `{0, …, n-1}`.
    pub fn new(n: usize) -> Self {
        Network {
            n,
            tail: Vec::new(),
            head: Vec::new(),
            cap: Vec::new(),
            orig_cap: Vec::new(),
            cost: Vec::new(),
            adj: vec![Vec::new(); n],
            level: vec![0; n],
            iter: vec![0; n],
        }
    }

    /// Number of vertices.
    pub fn num_vertices(&self) -> usize {
        self.n
    }

    /// Number of user-added (forward) arcs.
    pub fn num_arcs(&self) -> usize {
        self.head.len() / 2
    }

    /// Add a directed arc `u -> v` with non-negative integer `capacity` and zero
    /// cost.  Returns the arc id (usable with [`Network::flow_on_arc`]).
    pub fn add_arc(&mut self, u: usize, v: usize, capacity: Integer) -> usize {
        self.add_arc_with_cost(u, v, capacity, Integer::zero())
    }

    /// Add a directed arc `u -> v` with `capacity` and per-unit `cost`.
    pub fn add_arc_with_cost(
        &mut self,
        u: usize,
        v: usize,
        capacity: Integer,
        cost: Integer,
    ) -> usize {
        assert!(u < self.n && v < self.n, "arc endpoint out of range");
        let fwd = self.head.len();
        // forward arc
        self.tail.push(u);
        self.head.push(v);
        self.cap.push(capacity.clone());
        self.orig_cap.push(capacity);
        self.cost.push(cost.clone());
        self.adj[u].push(fwd);
        // reverse (residual) arc
        self.tail.push(v);
        self.head.push(u);
        self.cap.push(Integer::zero());
        self.orig_cap.push(Integer::zero());
        self.cost.push(Integer::zero() - cost);
        self.adj[v].push(fwd + 1);
        fwd
    }

    /// Build a network on `n` vertices from `(u, v, capacity)` triples.
    pub fn from_arcs(n: usize, arcs: &[(usize, usize, Integer)]) -> Self {
        let mut net = Network::new(n);
        for (u, v, c) in arcs {
            net.add_arc(*u, *v, c.clone());
        }
        net
    }

    /// Reset all residual capacities to the original capacities (undo any flow).
    fn reset(&mut self) {
        for i in 0..self.cap.len() {
            self.cap[i] = self.orig_cap[i].clone();
        }
    }

    /// The flow currently carried on forward arc `arc_id` (original capacity
    /// minus residual capacity).  Valid after a flow computation; zero after
    /// [`Network::reset`] or before any flow is run.  Mirrors MAGMA `Flow(e)`.
    pub fn flow_on_arc(&self, arc_id: usize) -> Integer {
        self.orig_cap[arc_id].clone() - self.cap[arc_id].clone()
    }

    /// The capacity of forward arc `arc_id` (MAGMA `Capacity(e)`).
    pub fn arc_capacity(&self, arc_id: usize) -> Integer {
        self.orig_cap[arc_id].clone()
    }

    // -- Dinic's algorithm ---------------------------------------------------

    fn bfs_level(&mut self, s: usize, t: usize) -> bool {
        for l in self.level.iter_mut() {
            *l = -1;
        }
        let mut queue = VecDeque::new();
        self.level[s] = 0;
        queue.push_back(s);
        while let Some(u) = queue.pop_front() {
            let arcs = self.adj[u].clone();
            for arc in arcs {
                let v = self.head[arc];
                if self.level[v] < 0 && self.cap[arc] > Integer::zero() {
                    self.level[v] = self.level[u] + 1;
                    queue.push_back(v);
                }
            }
        }
        self.level[t] >= 0
    }

    fn dfs_augment(&mut self, u: usize, t: usize, pushed: Integer) -> Integer {
        if u == t {
            return pushed;
        }
        while self.iter[u] < self.adj[u].len() {
            let arc = self.adj[u][self.iter[u]];
            let v = self.head[arc];
            if self.cap[arc] > Integer::zero() && self.level[v] == self.level[u] + 1 {
                let d = if pushed <= self.cap[arc] {
                    pushed.clone()
                } else {
                    self.cap[arc].clone()
                };
                let res = self.dfs_augment(v, t, d);
                if res > Integer::zero() {
                    self.cap[arc] = self.cap[arc].clone() - res.clone();
                    self.cap[arc ^ 1] = self.cap[arc ^ 1].clone() + res.clone();
                    return res;
                }
            }
            self.iter[u] += 1;
        }
        Integer::zero()
    }

    /// Run Dinic's algorithm from `s` to `t`, leaving the residual network in
    /// place, and return the value of the maximum flow.
    fn dinic(&mut self, s: usize, t: usize) -> Integer {
        self.reset();
        if s == t {
            return Integer::zero();
        }
        // Upper bound for a single augmenting push.
        let mut bound = Integer::zero();
        for a in (0..self.orig_cap.len()).step_by(2) {
            bound = bound + self.orig_cap[a].clone();
        }
        let mut flow = Integer::zero();
        while self.bfs_level(s, t) {
            for it in self.iter.iter_mut() {
                *it = 0;
            }
            loop {
                let f = self.dfs_augment(s, t, bound.clone());
                if f.is_zero() {
                    break;
                }
                flow = flow + f;
            }
        }
        flow
    }

    /// `MaximumFlow(s, t)` — the maximum flow value from source `s` to sink `t`
    /// (Dinic's algorithm).  After the call, [`Network::flow_on_arc`] reports the
    /// flow assigned to each arc.
    pub fn maximum_flow(&mut self, s: usize, t: usize) -> Integer {
        self.dinic(s, t)
    }

    /// `MinimumCut(s, t)` — returns `(S, F)` where `S ⊆ V` (with `s ∈ S`,
    /// `t ∉ S`) defines a minimum `s`–`t` cut and `F` is its capacity (equal to
    /// the maximum flow, by max-flow/min-cut).  `S` is the set of vertices
    /// reachable from `s` in the residual network.
    pub fn minimum_cut(&mut self, s: usize, t: usize) -> (Vec<usize>, Integer) {
        let f = self.dinic(s, t);
        // Vertices reachable from s along residual arcs with positive capacity.
        let mut reachable = vec![false; self.n];
        let mut stack = vec![s];
        reachable[s] = true;
        while let Some(u) = stack.pop() {
            for &arc in &self.adj[u] {
                let v = self.head[arc];
                if !reachable[v] && self.cap[arc] > Integer::zero() {
                    reachable[v] = true;
                    stack.push(v);
                }
            }
        }
        let cut: Vec<usize> = (0..self.n).filter(|&v| reachable[v]).collect();
        (cut, f)
    }

    /// The capacity of the cut `(S, V\S)`: the sum of original capacities of
    /// arcs from `S` to its complement.
    pub fn cut_capacity(&self, s_set: &[usize]) -> Integer {
        let mut in_s = vec![false; self.n];
        for &v in s_set {
            in_s[v] = true;
        }
        let mut total = Integer::zero();
        for a in (0..self.head.len()).step_by(2) {
            if in_s[self.tail[a]] && !in_s[self.head[a]] {
                total = total + self.orig_cap[a].clone();
            }
        }
        total
    }

    // -- Minimum-cost maximum flow (successive shortest paths / SPFA) ---------

    /// Minimum-cost maximum flow from `s` to `t`: returns `(flow, cost)` where
    /// `flow` is the maximum flow value and `cost` is the minimum total cost
    /// achieving it.  Uses successive shortest augmenting paths found by SPFA
    /// (Bellman–Ford queue), which tolerates the negative residual costs.
    ///
    /// Requires the network to have no negative-cost cycle (true for
    /// non-negative arc costs, the MAGMA setting).
    pub fn min_cost_max_flow(&mut self, s: usize, t: usize) -> (Integer, Integer) {
        self.reset();
        let mut flow = Integer::zero();
        let mut total_cost = Integer::zero();
        if s == t {
            return (flow, total_cost);
        }

        loop {
            // SPFA: shortest path by cost from s in the residual graph.
            let mut dist: Vec<Option<Integer>> = vec![None; self.n];
            let mut in_queue = vec![false; self.n];
            let mut prev_arc: Vec<Option<usize>> = vec![None; self.n];
            dist[s] = Some(Integer::zero());
            let mut queue = VecDeque::new();
            queue.push_back(s);
            in_queue[s] = true;

            while let Some(u) = queue.pop_front() {
                in_queue[u] = false;
                let du = dist[u].clone().unwrap();
                let arcs = self.adj[u].clone();
                for arc in arcs {
                    if self.cap[arc] <= Integer::zero() {
                        continue;
                    }
                    let v = self.head[arc];
                    let nd = du.clone() + self.cost[arc].clone();
                    let better = match &dist[v] {
                        None => true,
                        Some(dv) => nd < *dv,
                    };
                    if better {
                        dist[v] = Some(nd);
                        prev_arc[v] = Some(arc);
                        if !in_queue[v] {
                            in_queue[v] = true;
                            queue.push_back(v);
                        }
                    }
                }
            }

            if dist[t].is_none() {
                break; // t unreachable: max flow achieved
            }

            // Bottleneck along the found path.
            let mut push: Option<Integer> = None;
            let mut v = t;
            while v != s {
                let arc = prev_arc[v].unwrap();
                push = Some(match push {
                    None => self.cap[arc].clone(),
                    Some(p) => {
                        if self.cap[arc] < p {
                            self.cap[arc].clone()
                        } else {
                            p
                        }
                    }
                });
                v = self.tail[arc];
            }
            let push = push.unwrap();

            // Apply the augmentation.
            let mut v = t;
            while v != s {
                let arc = prev_arc[v].unwrap();
                self.cap[arc] = self.cap[arc].clone() - push.clone();
                self.cap[arc ^ 1] = self.cap[arc ^ 1].clone() + push.clone();
                v = self.tail[arc];
            }

            total_cost = total_cost + push.clone() * dist[t].clone().unwrap();
            flow = flow + push;
        }

        (flow, total_cost)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i(x: i64) -> Integer {
        Integer::from(x)
    }

    #[test]
    fn clrs_max_flow_is_23() {
        // Classic CLRS max-flow instance; s = 0, t = 5, max flow = 23.
        let mut net = Network::new(6);
        net.add_arc(0, 1, i(16));
        net.add_arc(0, 2, i(13));
        net.add_arc(1, 2, i(10));
        net.add_arc(2, 1, i(4));
        net.add_arc(1, 3, i(12));
        net.add_arc(3, 2, i(9));
        net.add_arc(2, 4, i(14));
        net.add_arc(4, 3, i(7));
        net.add_arc(3, 5, i(20));
        net.add_arc(4, 5, i(4));
        assert_eq!(net.maximum_flow(0, 5), i(23));
    }

    #[test]
    fn clrs_min_cut_capacity_equals_flow() {
        let mut net = Network::new(6);
        net.add_arc(0, 1, i(16));
        net.add_arc(0, 2, i(13));
        net.add_arc(1, 2, i(10));
        net.add_arc(2, 1, i(4));
        net.add_arc(1, 3, i(12));
        net.add_arc(3, 2, i(9));
        net.add_arc(2, 4, i(14));
        net.add_arc(4, 3, i(7));
        net.add_arc(3, 5, i(20));
        net.add_arc(4, 5, i(4));
        let (s_set, f) = net.minimum_cut(0, 5);
        assert_eq!(f, i(23));
        // Max-flow/min-cut: the cut capacity equals the flow value.
        assert_eq!(net.cut_capacity(&s_set), i(23));
        assert!(s_set.contains(&0));
        assert!(!s_set.contains(&5));
    }

    #[test]
    fn flow_conservation_and_arc_flow() {
        // Single path 0->1->2, cap 5 then 3 -> max flow 3.
        let mut net = Network::new(3);
        let a = net.add_arc(0, 1, i(5));
        let b = net.add_arc(1, 2, i(3));
        assert_eq!(net.maximum_flow(0, 2), i(3));
        assert_eq!(net.flow_on_arc(a), i(3));
        assert_eq!(net.flow_on_arc(b), i(3));
    }

    #[test]
    fn bipartite_matching_via_flow() {
        // Left {0,1}, right {2,3}, source 4, sink 5.
        // Allowed pairs: 0-2, 0-3, 1-2 -> maximum matching size 2.
        let mut net = Network::new(6);
        net.add_arc(4, 0, i(1));
        net.add_arc(4, 1, i(1));
        net.add_arc(0, 2, i(1));
        net.add_arc(0, 3, i(1));
        net.add_arc(1, 2, i(1));
        net.add_arc(2, 5, i(1));
        net.add_arc(3, 5, i(1));
        assert_eq!(net.maximum_flow(4, 5), i(2));
    }

    #[test]
    fn min_cost_max_flow_two_paths() {
        // Two disjoint s->t paths, caps 1 each, costs 2 and 6.
        // Max flow = 2, minimum cost = 8.
        let mut net = Network::new(4);
        net.add_arc_with_cost(0, 1, i(1), i(1));
        net.add_arc_with_cost(1, 3, i(1), i(1)); // path 0-1-3 cost 2
        net.add_arc_with_cost(0, 2, i(1), i(3));
        net.add_arc_with_cost(2, 3, i(1), i(3)); // path 0-2-3 cost 6
        let (flow, cost) = net.min_cost_max_flow(0, 3);
        assert_eq!(flow, i(2));
        assert_eq!(cost, i(8));
    }

    #[test]
    fn min_cost_prefers_cheaper_path_at_low_flow() {
        // Cheap path cap 1 (cost 1 total), expensive path cap 1 (cost 10).
        // If we only need flow 1, cost should be 1 (use the cheap path first).
        let mut net = Network::new(3);
        // 0->2 direct cheap cap1 cost1; 0->1->2 via cost path
        net.add_arc_with_cost(0, 2, i(1), i(1));
        net.add_arc_with_cost(0, 1, i(1), i(5));
        net.add_arc_with_cost(1, 2, i(1), i(5));
        let (flow, cost) = net.min_cost_max_flow(0, 2);
        // Max flow is 2 (both paths), min cost = 1 + 10 = 11.
        assert_eq!(flow, i(2));
        assert_eq!(cost, i(11));
    }

    #[test]
    fn no_flow_when_disconnected() {
        let mut net = Network::new(4);
        net.add_arc(0, 1, i(5));
        // 2->3 separate; no path 0->3
        net.add_arc(2, 3, i(5));
        assert_eq!(net.maximum_flow(0, 3), i(0));
    }
}
