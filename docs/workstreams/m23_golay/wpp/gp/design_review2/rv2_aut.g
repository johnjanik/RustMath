t := CharacterTable("M23");;
A := AutomorphismsOfTable(t);;
e := Exponent(t);;
ks := Filtered([2..2000], k -> IsPrime(k) and Gcd(k, e) = 1);;
gal := Group(List(ks, k -> PermList(PowerMap(t, k))));;
Print("exponent ", e, " |Aut| ", Size(A), " |Galois| ", Size(gal), " equal ", A = gal, "\n");
QUIT;
