intrinsic Foo(j::FldFinElt, success::BoolElt) -> BoolElt
{Doc.}
    if not IsField(j) then
        return false;
    elif success then
    // checking j in GF(p) span
        return true;
    elif Degree(j) eq 1 then
        return false;
    else
        return false;
    end if;
end intrinsic;
