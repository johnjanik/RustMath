intrinsic Foo(x::RngIntElt) -> BoolElt
{Doc.}
    if not assigned x then
        return false;
    end if;
    if assigned y then
        return true;
    end if;
    return false;
end intrinsic;
