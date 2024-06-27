# lec3Notes

general eqn:
    line eqn = y=wx+b
    MSE = mean square error = 1/n  segma (yi -y'i)^2
    MAE = mean absolute error = 1/n segma |yi-y'i|
    J(w) = yp-yr
    J(w,b)=1/2m seg (yi -y'i)^2

notes:
    the method is : gradiant descent

the flow of process:
    1 assume any values for w and b
    2 draw line
    3 calaulate J
    4 calculate new w and b
    5 draw line 
    6 calc J 

notes:
    w-new = w-old - alfa dJ/dw
    b-new = b-old - alfa dJ/db
    dJ/dw =2/n seg yp-yr
    dJ/db =2/n seg yp-yr
    alfa = selected from (alfa < 1)


implementation notes:

    func = cost func => cost
    declare = w,b init 
    func = plot 
    func = w,b new >> d/d == descent func 
    func = enter data 