intrinsic Foo(bas::SeqEnum) -> BoolElt
{Doc.}
    vtime MWSieve: flag, unsat := CheckSaturationAtPrimes(bas, PrimesUpTo(SmoothBound));
    vtime MWSieve, 2: x := Compute(bas);
    return flag;
end intrinsic;
