for i in   [1..10] do
  if i eq 5 then break; end if;
  if i  eq  7 then    continue; end if;
        Append(~L,i);
  end for;
