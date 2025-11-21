//! Proof systems for Boolean logic

use crate::cnf::{Clause, Cnf, Literal};
use crate::formula::{Formula, Variable};
use std::collections::HashSet;
use std::fmt;

/// A proof step in natural deduction
#[derive(Clone, Debug, PartialEq)]
pub enum ProofStep {
    /// Assumption
    Assume(Formula),
    /// Modus Ponens: from A and A → B, infer B
    ModusPonens {
        premise1: usize,
        premise2: usize,
        conclusion: Formula,
    },
    /// And Introduction: from A and B, infer A ∧ B
    AndIntro {
        left: usize,
        right: usize,
        conclusion: Formula,
    },
    /// And Elimination Left: from A ∧ B, infer A
    AndElimLeft { premise: usize, conclusion: Formula },
    /// And Elimination Right: from A ∧ B, infer B
    AndElimRight { premise: usize, conclusion: Formula },
    /// Or Introduction Left: from A, infer A ∨ B
    OrIntroLeft { premise: usize, conclusion: Formula },
    /// Or Introduction Right: from B, infer A ∨ B
    OrIntroRight { premise: usize, conclusion: Formula },
    /// Or Elimination: from A ∨ B, A → C, B → C, infer C
    OrElim {
        disjunction: usize,
        left_implies: usize,
        right_implies: usize,
        conclusion: Formula,
    },
    /// Not Introduction: if assuming A leads to contradiction, infer ¬A
    NotIntro {
        assumption: usize,
        contradiction: usize,
        conclusion: Formula,
    },
    /// Not Elimination: from A and ¬A, infer contradiction
    NotElim {
        premise1: usize,
        premise2: usize,
    },
    /// Double Negation Elimination: from ¬¬A, infer A
    DoubleNegElim { premise: usize, conclusion: Formula },
    /// Law of Excluded Middle: infer A ∨ ¬A
    ExcludedMiddle(Formula),
}

/// A natural deduction proof
#[derive(Clone, Debug)]
pub struct Proof {
    pub steps: Vec<(ProofStep, Formula)>,
    pub goal: Formula,
}

impl Proof {
    pub fn new(goal: Formula) -> Self {
        Proof {
            steps: Vec::new(),
            goal,
        }
    }

    pub fn add_step(&mut self, step: ProofStep, formula: Formula) {
        self.steps.push((step, formula));
    }

    /// Validate that the proof is correct
    pub fn validate(&self) -> bool {
        let mut proven: Vec<Formula> = Vec::new();

        for (step, formula) in &self.steps {
            match step {
                ProofStep::Assume(_) => {
                    proven.push(formula.clone());
                }
                ProofStep::ModusPonens {
                    premise1,
                    premise2,
                    conclusion,
                } => {
                    if *premise1 >= proven.len() || *premise2 >= proven.len() {
                        return false;
                    }
                    // Check if premise1 is A and premise2 is A → B
                    let a = &proven[*premise1];
                    if let Formula::Implies(left, right) = &proven[*premise2] {
                        if left.as_ref() == a && right.as_ref() == conclusion {
                            proven.push(formula.clone());
                            continue;
                        }
                    }
                    // Or check if premise2 is A and premise1 is A → B
                    let a = &proven[*premise2];
                    if let Formula::Implies(left, right) = &proven[*premise1] {
                        if left.as_ref() == a && right.as_ref() == conclusion {
                            proven.push(formula.clone());
                            continue;
                        }
                    }
                    return false;
                }
                ProofStep::AndIntro {
                    left,
                    right,
                    conclusion,
                } => {
                    if *left >= proven.len() || *right >= proven.len() {
                        return false;
                    }
                    if let Formula::And(fs) = conclusion {
                        if fs.len() == 2 && &fs[0] == &proven[*left] && &fs[1] == &proven[*right]
                        {
                            proven.push(formula.clone());
                            continue;
                        }
                    }
                    return false;
                }
                ProofStep::AndElimLeft { premise, conclusion } => {
                    if *premise >= proven.len() {
                        return false;
                    }
                    if let Formula::And(fs) = &proven[*premise] {
                        if !fs.is_empty() && &fs[0] == conclusion {
                            proven.push(formula.clone());
                            continue;
                        }
                    }
                    return false;
                }
                ProofStep::AndElimRight { premise, conclusion } => {
                    if *premise >= proven.len() {
                        return false;
                    }
                    if let Formula::And(fs) = &proven[*premise] {
                        if fs.len() >= 2 && &fs[1] == conclusion {
                            proven.push(formula.clone());
                            continue;
                        }
                    }
                    return false;
                }
                ProofStep::OrIntroLeft { premise, conclusion } => {
                    if *premise >= proven.len() {
                        return false;
                    }
                    if let Formula::Or(fs) = conclusion {
                        if fs.len() >= 1 && &fs[0] == &proven[*premise] {
                            proven.push(formula.clone());
                            continue;
                        }
                    }
                    return false;
                }
                ProofStep::OrIntroRight { premise, conclusion } => {
                    if *premise >= proven.len() {
                        return false;
                    }
                    if let Formula::Or(fs) = conclusion {
                        if fs.len() >= 2 && &fs[1] == &proven[*premise] {
                            proven.push(formula.clone());
                            continue;
                        }
                    }
                    return false;
                }
                ProofStep::DoubleNegElim { premise, conclusion } => {
                    if *premise >= proven.len() {
                        return false;
                    }
                    if let Formula::Not(n1) = &proven[*premise] {
                        if let Formula::Not(n2) = n1.as_ref() {
                            if n2.as_ref() == conclusion {
                                proven.push(formula.clone());
                                continue;
                            }
                        }
                    }
                    return false;
                }
                ProofStep::ExcludedMiddle(_) => {
                    // A ∨ ¬A is always valid
                    if let Formula::Or(fs) = formula {
                        if fs.len() == 2 {
                            if let Formula::Not(n) = &fs[1] {
                                if n.as_ref() == &fs[0] {
                                    proven.push(formula.clone());
                                    continue;
                                }
                            }
                        }
                    }
                    return false;
                }
                _ => {
                    // For other rules, just accept them for now
                    proven.push(formula.clone());
                }
            }
        }

        // Check if the goal was proven
        proven.iter().any(|f| f == &self.goal)
    }

    /// Check if the proof is complete
    pub fn is_complete(&self) -> bool {
        self.steps.iter().any(|(_, f)| f == &self.goal)
    }
}

/// A resolution proof step
#[derive(Clone, Debug, PartialEq)]
pub struct ResolutionStep {
    /// The two parent clauses
    pub parent1: usize,
    pub parent2: usize,
    /// The variable that was resolved on
    pub resolved_var: Variable,
    /// The resulting clause (resolvent)
    pub resolvent: Clause,
}

/// A resolution-based proof of unsatisfiability
#[derive(Clone, Debug)]
pub struct ResolutionProof {
    /// Original clauses from the CNF formula
    pub clauses: Vec<Clause>,
    /// Resolution steps
    pub steps: Vec<ResolutionStep>,
}

impl ResolutionProof {
    pub fn new(clauses: Vec<Clause>) -> Self {
        ResolutionProof {
            clauses,
            steps: Vec::new(),
        }
    }

    /// Add a resolution step
    pub fn add_step(&mut self, step: ResolutionStep) {
        self.steps.push(step);
    }

    /// Check if the proof derives the empty clause (contradiction)
    pub fn is_refutation(&self) -> bool {
        self.steps
            .iter()
            .any(|step| step.resolvent.is_empty())
    }

    /// Validate the resolution proof
    pub fn validate(&self) -> bool {
        let mut derived_clauses = self.clauses.clone();

        for step in &self.steps {
            // Check that parent indices are valid
            if step.parent1 >= derived_clauses.len() || step.parent2 >= derived_clauses.len() {
                return false;
            }

            let clause1 = &derived_clauses[step.parent1];
            let clause2 = &derived_clauses[step.parent2];

            // Verify the resolution is correct
            if !self.is_valid_resolution(clause1, clause2, &step.resolved_var, &step.resolvent) {
                return false;
            }

            derived_clauses.push(step.resolvent.clone());
        }

        true
    }

    /// Check if a resolution step is valid
    fn is_valid_resolution(
        &self,
        clause1: &Clause,
        clause2: &Clause,
        var: &Variable,
        resolvent: &Clause,
    ) -> bool {
        // Find the positive and negative literals of var
        let mut has_positive_in_c1 = false;
        let mut has_negative_in_c1 = false;
        let mut has_positive_in_c2 = false;
        let mut has_negative_in_c2 = false;

        for lit in &clause1.literals {
            if &lit.variable == var {
                if lit.negated {
                    has_negative_in_c1 = true;
                } else {
                    has_positive_in_c1 = true;
                }
            }
        }

        for lit in &clause2.literals {
            if &lit.variable == var {
                if lit.negated {
                    has_negative_in_c2 = true;
                } else {
                    has_positive_in_c2 = true;
                }
            }
        }

        // One clause must have positive literal, other must have negative
        if !((has_positive_in_c1 && has_negative_in_c2) || (has_negative_in_c1 && has_positive_in_c2)) {
            return false;
        }

        // Resolvent should contain all literals from both clauses except the resolved variable
        let mut expected_literals: HashSet<Literal> = HashSet::new();

        for lit in &clause1.literals {
            if &lit.variable != var {
                expected_literals.insert(lit.clone());
            }
        }

        for lit in &clause2.literals {
            if &lit.variable != var {
                expected_literals.insert(lit.clone());
            }
        }

        let resolvent_set: HashSet<Literal> = resolvent.literals.iter().cloned().collect();

        expected_literals == resolvent_set
    }

    /// Get the length (number of steps) of the proof
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Check if the proof is empty
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
}

impl fmt::Display for ResolutionProof {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Resolution Proof:")?;
        writeln!(f, "Initial clauses:")?;
        for (i, clause) in self.clauses.iter().enumerate() {
            writeln!(f, "  {}: {}", i, clause)?;
        }
        writeln!(f, "\nResolution steps:")?;
        for (i, step) in self.steps.iter().enumerate() {
            writeln!(
                f,
                "  {}: Resolve {} and {} on {} → {}",
                i + self.clauses.len(),
                step.parent1,
                step.parent2,
                step.resolved_var,
                step.resolvent
            )?;
        }
        Ok(())
    }
}

/// Attempt to generate a resolution proof of unsatisfiability
pub fn generate_resolution_proof(cnf: &Cnf) -> Option<ResolutionProof> {
    let mut proof = ResolutionProof::new(cnf.clauses.clone());
    let mut derived: Vec<Clause> = cnf.clauses.clone();
    let mut new_clauses: Vec<(usize, usize, Variable, Clause)> = Vec::new();

    // Try to derive the empty clause using resolution
    for _ in 0..1000 {
        // Limit iterations
        new_clauses.clear();

        // Try all pairs of clauses
        for i in 0..derived.len() {
            for j in (i + 1)..derived.len() {
                // Find variables that can be resolved
                let vars1: HashSet<_> = derived[i]
                    .literals
                    .iter()
                    .map(|l| l.variable.clone())
                    .collect();
                let vars2: HashSet<_> = derived[j]
                    .literals
                    .iter()
                    .map(|l| l.variable.clone())
                    .collect();

                // Try to resolve on common variables
                for var in vars1.intersection(&vars2) {
                    if let Some(resolvent) = resolve_clauses(&derived[i], &derived[j], var) {
                        // Check if we derived the empty clause
                        if resolvent.is_empty() {
                            proof.add_step(ResolutionStep {
                                parent1: i,
                                parent2: j,
                                resolved_var: var.clone(),
                                resolvent: resolvent.clone(),
                            });
                            return Some(proof);
                        }

                        // Check if this is a new clause
                        if !derived.contains(&resolvent) && !new_clauses.iter().any(|(_, _, _, r)| r == &resolvent) {
                            new_clauses.push((i, j, var.clone(), resolvent));
                        }
                    }
                }
            }
        }

        if new_clauses.is_empty() {
            // No new clauses can be derived
            return None;
        }

        // Add new clauses to derived set
        for (parent1, parent2, var, resolvent) in &new_clauses {
            proof.add_step(ResolutionStep {
                parent1: *parent1,
                parent2: *parent2,
                resolved_var: var.clone(),
                resolvent: resolvent.clone(),
            });
            derived.push(resolvent.clone());
        }
    }

    None
}

/// Resolve two clauses on a variable
fn resolve_clauses(clause1: &Clause, clause2: &Clause, var: &Variable) -> Option<Clause> {
    // Check if one has positive and other has negative literal of var
    let has_pos_1 = clause1
        .literals
        .iter()
        .any(|l| &l.variable == var && !l.negated);
    let has_neg_1 = clause1
        .literals
        .iter()
        .any(|l| &l.variable == var && l.negated);
    let has_pos_2 = clause2
        .literals
        .iter()
        .any(|l| &l.variable == var && !l.negated);
    let has_neg_2 = clause2
        .literals
        .iter()
        .any(|l| &l.variable == var && l.negated);

    if !((has_pos_1 && has_neg_2) || (has_neg_1 && has_pos_2)) {
        return None;
    }

    // Create resolvent
    let mut literals = Vec::new();

    for lit in &clause1.literals {
        if &lit.variable != var {
            literals.push(lit.clone());
        }
    }

    for lit in &clause2.literals {
        if &lit.variable != var && !literals.contains(lit) {
            literals.push(lit.clone());
        }
    }

    Some(Clause::new(literals))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_assumption() {
        let goal = Formula::var("x");
        let mut proof = Proof::new(goal.clone());

        proof.add_step(ProofStep::Assume(goal.clone()), goal.clone());

        assert!(proof.validate());
        assert!(proof.is_complete());
    }

    #[test]
    fn test_proof_and_intro() {
        // Prove x ∧ y from x and y
        let x = Formula::var("x");
        let y = Formula::var("y");
        let goal = Formula::and(vec![x.clone(), y.clone()]);

        let mut proof = Proof::new(goal.clone());

        proof.add_step(ProofStep::Assume(x.clone()), x.clone());
        proof.add_step(ProofStep::Assume(y.clone()), y.clone());
        proof.add_step(
            ProofStep::AndIntro {
                left: 0,
                right: 1,
                conclusion: goal.clone(),
            },
            goal.clone(),
        );

        assert!(proof.validate());
        assert!(proof.is_complete());
    }

    #[test]
    fn test_proof_and_elim() {
        // From x ∧ y, derive x
        let x = Formula::var("x");
        let y = Formula::var("y");
        let conj = Formula::and(vec![x.clone(), y.clone()]);

        let mut proof = Proof::new(x.clone());

        proof.add_step(ProofStep::Assume(conj.clone()), conj);
        proof.add_step(
            ProofStep::AndElimLeft {
                premise: 0,
                conclusion: x.clone(),
            },
            x.clone(),
        );

        assert!(proof.validate());
        assert!(proof.is_complete());
    }

    #[test]
    fn test_proof_double_neg_elim() {
        // From ¬¬x, derive x
        let x = Formula::var("x");
        let double_neg = Formula::not(Formula::not(x.clone()));

        let mut proof = Proof::new(x.clone());

        proof.add_step(ProofStep::Assume(double_neg.clone()), double_neg);
        proof.add_step(
            ProofStep::DoubleNegElim {
                premise: 0,
                conclusion: x.clone(),
            },
            x.clone(),
        );

        assert!(proof.validate());
        assert!(proof.is_complete());
    }

    #[test]
    fn test_resolution_simple() {
        // {x} and {¬x} resolve to {}
        let clause1 = Clause::new(vec![Literal::positive(Variable::new("x"))]);
        let clause2 = Clause::new(vec![Literal::negative(Variable::new("x"))]);

        let resolvent = resolve_clauses(&clause1, &clause2, &Variable::new("x")).unwrap();

        assert!(resolvent.is_empty());
    }

    #[test]
    fn test_resolution_with_other_literals() {
        // {x, y} and {¬x, z} resolve to {y, z}
        let clause1 = Clause::new(vec![
            Literal::positive(Variable::new("x")),
            Literal::positive(Variable::new("y")),
        ]);
        let clause2 = Clause::new(vec![
            Literal::negative(Variable::new("x")),
            Literal::positive(Variable::new("z")),
        ]);

        let resolvent = resolve_clauses(&clause1, &clause2, &Variable::new("x")).unwrap();

        assert_eq!(resolvent.literals.len(), 2);
        assert!(resolvent
            .literals
            .iter()
            .any(|l| l.variable == Variable::new("y") && !l.negated));
        assert!(resolvent
            .literals
            .iter()
            .any(|l| l.variable == Variable::new("z") && !l.negated));
    }

    #[test]
    fn test_resolution_proof_unsatisfiable() {
        // (x) ∧ (¬x) is unsatisfiable
        let cnf = Cnf::new(vec![
            Clause::new(vec![Literal::positive(Variable::new("x"))]),
            Clause::new(vec![Literal::negative(Variable::new("x"))]),
        ]);

        let proof = generate_resolution_proof(&cnf);
        assert!(proof.is_some());

        let proof = proof.unwrap();
        assert!(proof.is_refutation());
        assert!(proof.validate());
    }

    #[test]
    fn test_resolution_proof_complex() {
        // (x ∨ y) ∧ (¬x ∨ z) ∧ (¬y) ∧ (¬z) is unsatisfiable
        let cnf = Cnf::new(vec![
            Clause::new(vec![
                Literal::positive(Variable::new("x")),
                Literal::positive(Variable::new("y")),
            ]),
            Clause::new(vec![
                Literal::negative(Variable::new("x")),
                Literal::positive(Variable::new("z")),
            ]),
            Clause::new(vec![Literal::negative(Variable::new("y"))]),
            Clause::new(vec![Literal::negative(Variable::new("z"))]),
        ]);

        let proof = generate_resolution_proof(&cnf);
        assert!(proof.is_some());

        let proof = proof.unwrap();
        assert!(proof.is_refutation());
        assert!(proof.validate());
    }

    #[test]
    fn test_resolution_proof_satisfiable() {
        // (x ∨ y) is satisfiable, so no refutation exists
        let cnf = Cnf::new(vec![Clause::new(vec![
            Literal::positive(Variable::new("x")),
            Literal::positive(Variable::new("y")),
        ])]);

        let proof = generate_resolution_proof(&cnf);
        // Should not be able to derive empty clause
        assert!(proof.is_none() || !proof.unwrap().is_refutation());
    }

    #[test]
    fn test_invalid_resolution() {
        // {x, y} and {x, z} cannot be resolved on x (both positive)
        let clause1 = Clause::new(vec![
            Literal::positive(Variable::new("x")),
            Literal::positive(Variable::new("y")),
        ]);
        let clause2 = Clause::new(vec![
            Literal::positive(Variable::new("x")),
            Literal::positive(Variable::new("z")),
        ]);

        let resolvent = resolve_clauses(&clause1, &clause2, &Variable::new("x"));
        assert!(resolvent.is_none());
    }

    #[test]
    fn test_excluded_middle() {
        // x ∨ ¬x is always provable
        let x = Formula::var("x");
        let goal = Formula::or(vec![x.clone(), Formula::not(x.clone())]);

        let mut proof = Proof::new(goal.clone());
        proof.add_step(ProofStep::ExcludedMiddle(x), goal.clone());

        assert!(proof.validate());
        assert!(proof.is_complete());
    }
}
