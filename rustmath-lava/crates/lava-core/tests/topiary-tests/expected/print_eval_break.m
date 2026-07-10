intrinsic Foo(x::RngIntElt) -> RngIntElt
{Doc.}
    print "hello";
    printf "value: %o\n", x;
    fprintf out, "log: %o\n", x;
    vprint Verbose: "info";
    vprintf Verbose, 1: "value: %o\n", x;
    y := eval Sprintf("Foo(%o)", x);
    for i in [1..10] do
        if i eq 5 then
            break;
        end if;
        if i eq 7 then
            continue;
        end if;
    end for;
    return y;
end intrinsic;
