intrinsic Foo(x::BoolElt) -> RngIntElt
{Doc.}
if not x then
// print "no";
else return 1; end if;
if x then  return 2; end if;
return 0;
end intrinsic;
