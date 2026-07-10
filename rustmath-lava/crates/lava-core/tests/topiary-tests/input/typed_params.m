procedure Foo(x::RngIntElt,y::SeqEnum)
   return [y[i]:i in [1..#y]];
end procedure;

procedure Bar(S::SetIndx,k::RngIntElt)
return S;
end procedure;
