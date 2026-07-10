intrinsic Foo(n::RngIntElt) -> MonStgElt
{Doc.}
s:=n eq 1 select"singular"else"plural";
   t :=n gt 0 select "positive"  else n lt 0 select  "negative"   else "zero";
error if not   IsPrime(n),"n must be prime";
   error if n eq 0,"n must be nonzero";
return s;
end intrinsic;
