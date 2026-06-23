intrinsic Foo(x::RngIntElt) -> BoolElt
{Doc.}
    assert x ge 0;
    assert2 forall{P : P in primes | IsIntegral(P)};
    assert3 IsPrime(x);
    return true;
end intrinsic;
