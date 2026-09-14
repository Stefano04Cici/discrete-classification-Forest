:- dynamic(prop/3).

carat_class(Carat, low) :- Carat < 0.5.
carat_class(Carat, medium) :- Carat >= 0.5, Carat < 1.0.
carat_class(Carat, high) :- Carat >= 1.0.

depth_class(Depth, low) :- Depth < 61.0.
depth_class(Depth, medium) :- Depth >= 61.0, Depth =< 62.5.
depth_class(Depth, high) :- Depth > 62.5.

table_class(Table, low) :- Table < 56.
table_class(Table, medium) :- Table >= 56, Table =< 58.
table_class(Table, high) :- Table > 58.

x_class(X, low) :- X < 5.0.
x_class(X, medium) :- X >= 5.0, X < 6.4.
x_class(X, high) :- X >= 6.4.

y_class(Y, low) :- Y < 5.0.
y_class(Y, medium) :- Y >= 5.0, Y < 6.4.
y_class(Y, high) :- Y >= 6.4.

z_class(Z, low) :- Z < 3.1.
z_class(Z, medium) :- Z >= 3.1, Z =< 3.9.
z_class(Z, high) :- Z > 3.9.

price_class(Price, low) :- Price < 1000.
price_class(Price, medium) :- Price >= 1000, Price =< 4000.
price_class(Price, high) :- Price > 4000.


prop(Diamond, carat_class, Class) :-
    prop(Diamond, carat, Carat),
    carat_class(Carat, Class).

prop(Diamond, depth_class, Class) :-
    prop(Diamond, depth, Depth),
    depth_class(Depth, Class).

prop(Diamond, table_class, Class) :-
    prop(Diamond, table, Table),
    table_class(Table, Class).

prop(Diamond, x_class, Class) :-
    prop(Diamond, x, X),
    x_class(X, Class).

prop(Diamond, y_class, Class) :-
    prop(Diamond, y, Y),
    y_class(Y, Class).

prop(Diamond, z_class, Class) :-
    prop(Diamond, z, Z),
    z_class(Z, Class).

prop(Diamond, price_class, Class) :-
    prop(Diamond, price, Price),
    price_class(Price, Class).











%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% GENERATED FACTS BELOW THIS LINE %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





