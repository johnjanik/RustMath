freeze;

intrinsic Foo(x::RngIntElt) -> RngIntElt
{Return x.}
    return x;
end intrinsic;
intrinsic Bar(y::RngIntElt) -> RngIntElt
{Return y.}
    return y;
end intrinsic;

function helper(z)
    return z + 1;
end function;
procedure stash(~T, k)
    Append(~T, k);
end procedure;
