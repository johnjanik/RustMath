intrinsic Foo() ->RngIntElt
{Doc.}
for u in   [1..10] do
   for k  in [1..5] do
   if k eq 3 then  break u;end if;
       if k eq 4 then continue u; end if;
        if k eq 2 then break;end if;
   if k eq 1 then continue;
       end if;
   end for;
end for;
   return 0;
end intrinsic;
