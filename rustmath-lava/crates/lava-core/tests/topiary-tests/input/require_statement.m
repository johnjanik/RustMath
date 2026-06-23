intrinsic Foo(n::RngIntElt,k::RngIntElt,C::Crv) -> BoolElt
{Doc.}
requirege n,1;
   requirerange k,1,10;
require IsAffine(C):"Curve must be affine";
   require n gt k:"n must exceed k";
return true;
end intrinsic;
