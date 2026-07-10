forward helper,helper2;
intrinsic Foo(C::Crv) -> Crv
{Doc.}
return Curve(A,RA!(g(f)/RA.1^m)) where m is Minimum([TotalDegree(m):m in Monomials(f)]) where f is Polynomial(C);
end intrinsic;


intrinsic Bar(x::RngIntElt) -> RngIntElt
{Doc.}
return y  where y:=x+1;
end intrinsic;
