intrinsic Foo() -> RngIntElt
{Doc.}
    read x;
    read y, "Enter Y: ";
    readi z;
    if z eq 0 then
        exit 0;
    end if;
    quit;
    return x;
end intrinsic;
