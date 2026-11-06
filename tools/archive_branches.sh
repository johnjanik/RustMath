#!/bin/sh
# Archive the 19 consolidated branches: tag each tip as archive/<branch>, push the tags,
# then delete the branches on origin. See docs/BRANCHES.md for why each one is safe.
# Run from a clone of github.com/johnjanik/RustMath with push rights:  sh tools/archive_branches.sh
set -eu
BRANCHES="belyi/u0-unblock-and-true-root-detector igp24-cli-galois integrate-lava-galois
magma-port/wave0-foundation magma-port/wave1-combinatorics magma-port/wave1-crypto
magma-port/wave1-finitefields magma-port/wave1-graphs magma-port/wave1-matrix
magma-port/wave1-monoids-automata magma-port/wave1-powerseries
magma-port/wave1-symmetricfunctions p0-mestre-tools p1-prime-ideals padic-factorization-port
phase1-factorization-over-z phase2-local-analysis phase3-discriminant-lll phase4-galois-narrowing"

git fetch origin --prune
for b in $BRANCHES; do
  git rev-parse --verify -q "refs/remotes/origin/$b" >/dev/null || { echo "skip (gone): $b"; continue; }
  git tag -a -f "archive/$b" -m "Archived branch $b (tip before deletion). See docs/BRANCHES.md." "origin/$b"
done
git push origin $(for b in $BRANCHES; do
  git rev-parse --verify -q "refs/tags/archive/$b" >/dev/null && echo "refs/tags/archive/$b"; done)
# delete a branch only if its archive tag on origin points at the same commit
for b in $BRANCHES; do
  tip=$(git rev-parse -q --verify "refs/remotes/origin/$b") || continue
  tagged=$(git ls-remote origin "refs/tags/archive/$b^{}" | cut -f1)
  if [ "$tip" = "$tagged" ]; then git push origin --delete "$b"; else echo "NOT deleting $b: tag mismatch"; fi
done
echo "Remaining remote branches:"; git ls-remote --heads origin | sed 's#.*refs/heads/##'
