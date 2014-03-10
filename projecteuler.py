def problem17():
    digitsdict = {
        1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six',
        7:'seven', 8:'eight', 9:'nine', 10:'ten', 11:'eleven', 12:'twelve',
        13:'thirteen', 14:'fourteen', 15:'fifteen', 16:'sixteen',
        17:'seventeen', 18:'eighteen', 19:'nineteen', 20:'twenty',
        30:'thirty', 40:'forty', 50:'fifty', 60:'sixty', 70:'seventy',
        80:'eighty', 90:'ninety'
        }

    def getones(n):
        return len(digitsdict[n])

    def gettens(n):
        if n < 20:
            return len(digitsdict[n])
        elif n % 10 == 0:
            return len(digitsdict[n])
        else:
            tens = (n/10)*10
            ones = n%10
            return len(digitsdict[tens]) + len(digitsdict[ones])

    def gethundreds(n):
        tot = 0
        if n % 100 == 0:
            return getones(n/100) + 7 #7 = len("hundred")
        else:
            hundreds = n/100
            remaining = n % 100
            if remaining < 10:
                return getones(n/100) + 10 + getones(remaining) #10 = len("hundred and")
            else:
                return getones(n/100) + 10 + gettens(remaining)
        

    tot = 11 #including "one thousand" here
    for n in range(1,1000):
        if n < 10:
            tot += getones(n)
        elif 10 <= n < 100:
            tot += gettens(n)
        else:
            tot += gethundreds(n)
    print tot
        
def problem18():
        
    f = open("problem67nums.txt", 'r')
    tree = []
    for line in f:
        tree = tree + [map(int,line.split())]

    """
    Recursive solution.
    """
    def max_route_sum(i, j):
        if i==len(tree)-1:
            return int(tree[i][j])

        left_route = max_route_sum(i+1, j)
        right_route = max_route_sum(i+1, j+1)
        if left_route >= right_route:
            return left_route + int(tree[i][j])
        else:
            return right_route + int(tree[i][j])


##    i=0
##    j=0
##    print max_route_sum(i, j)
    

    """
    Iterative Solution. Rolls the pyramid up from the bottom, keeping the
    largest sum from the pervious row + current row.
    """
    for row in xrange(1,len(tree)):
            for col in xrange(len(tree[row])):
                if col == 0: #position is on far left of pyramid
                    tree[row][col] += tree[row-1][0] 
                elif col == len(tree[row])-1: #position on far right of pyramid
                    tree[row][col] += tree[row-1][col-1]
                else: #position can go either UR or UL
                    tree[row][col] += max(tree[row-1][col-1],tree[row-1][col])

    print max(tree[len(tree)-1])

def problem20():
    from math import factorial
    
    def sumofdigits(n):
        tot = 0
        for digit in str(n):
            tot += int(digit)
        return tot

    sumofnine = sumofdigits(factorial(100))
    print sumofnine

def problem21():
    import time
    from math import sqrt, ceil
    def getfactors(n):
        factors = [1]
        for i in range(2,int(ceil(sqrt(n)))+1):
            if n % i == 0:
                factors.append(i)
                factors.append(n/i)
        #perfect square roots are added twice to factors, remove one.
        if sqrt(n) in factors:
            factors.remove(sqrt(n))
            
        return factors

    def d(n):
        factors = getfactors(n)
        return sum(factors)

    start = time.time()
    for a in range(2,10000):
        b = d(a)
        if d(b) == a and a != b:
                print a, b
    end = time.time()
    dt = end - start*1.0
    print dt

def problem22():
    f = open('names.txt', 'r')
    lines = f.read().split(',')
    nameslist = sorted(lines)

    letters = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
               'N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    index = range(1,27)
    pointsdict = dict(zip(letters,index))
        

    total = 0
    index = 1
    for name in nameslist:
        score = 0
        for letter in name.strip('"'):
            score += pointsdict[letter]
        score *= index
        total += score
        index +=1
    print total    

def problem75():
    def gcd(a,b):
        """ the euclidean algorithm
            calculates the greatest common divisor of a and b
        """
        while a:
                a, b = b%a, a
        return b

    def evenodd(a,b):
        """
        An even plus an odd is always an odd, any other combination results in
        an even number.
        """
        n = a + b
        if (n % 2 == 0):
            return False
        return True
        
    def generateprimatives(limit):
        """generates all pythagorean triplet primatives that are of the form
        m(a,b,c) up to limit.

        (a,b,c) is primative IFF gcd(m,n) = 1 and (m,n) has exactly one even
        and one odd.

        m is upper bounded by sqrt((limit)/2) since c = m^2 + n^2
        """
        outlist = []
        from math import sqrt, ceil
        upper_limit = int(ceil(sqrt(limit/2)))+1
        for m in xrange(2,upper_limit):
            for n in xrange(1,m):
                if(evenodd(m,n) and gcd(m,n) == 1):
                    a = m*m - n*n
                    b = 2*m*n
                    c = m*m + n*n
                    outlist.append((a,b,c))
        return outlist

    def getsolns(n, primlist):
        limit = n
        mlimit = limit/2
        numsolns = 0
        for triplet in primlist:
            a,b,c = triplet[0],triplet[1],triplet[2]
            if(a+b+c > n):
                break
            for m in xrange(1,mlimit):
                if( (m*a) + (m*b) + (m*c) == limit):
                   numsolns +=1
                   break
        return numsolns

    upper_limit = 1500000
    count = 0
    L = 12
    print "Generating list of primatives." 
    primlist = generateprimatives(upper_limit)
    print "Primative list generated. Length: ", len(primlist)
    while(L <= upper_limit):
        print L
        num_solns = getsolns(L,primlist)
        if num_solns == 1:
            count +=1
        L+=1
    print count

def problem75bettersoln():
    """There were a lot of values of L that had 0 solutions. These all had to be
    checked in L/2. Thats a lot of useless searching.

    It is better, therefore, to think of this problem in terms of m and n.

    Finds an m and n that make a primative. (m+n is even and gcd(m,n) = 1)

    Creates that primative, then adds it together to find solution L.

    1. Records the solution in triplets table at index L. Checks if there is
    currently only one solution for L, if there is, add one to count.
    If triplets table at index L is greater than one, more than one solution
    exists for length L, remove that L from count.

    2. Finds the next solution of the form 2(a,b,c) by adding a,b,c.

    Repeat (1,2) until some constant C times our primative is greater than
    the upper limit.

    Repeat for all primatives such that (a+b+c) < limit.
    
    """
    from math import sqrt
    
    def gcd(a,b):
        """ the euclidean algorithm """
        while a:
                a, b = b%a, a
        return b
    
    limit = 1500000
    triplets = [0] * 1500001
    
    count = 0
    mlimit = int(sqrt(limit/2))


    for m in xrange(2,mlimit):
        for n in xrange(1,m):
            if( ((n+m)%2) == 1 and gcd(n,m) == 1):
                a = m * m - n * n
                b = 2 * m * n
                c = m * m + n * n
                p = a + b + c
                #this loop finds all possible solutions for:
                #L = (a,b,c) for p <= L <= limit (some m(a+b+c) )
                while(p <= limit):
                    #soln exists for primative (a,b,c) = p
                    triplets[p] += 1
                    if(triplets[p] == 1): #only care about only one soln
                        count += 1
                    if(triplets[p] == 2):
                        #more than one soln exists, remove it from count
                        count -= 1
                    #find next soln for multiple of (a,b,c)
                    p += a+b+c

    print count

def problem78():
    import sys
    sys.setrecursionlimit(100000)
    memo = {}

    def p(k,n):
        if n == 0 or k < 0: return 0
        elif k == 0: return 1
        if not (k,n) in memo:
            memo[(k,n)] = p(k,n-1) + p(k-n,n)
        return memo[(k,n)]

    n = 100
    while(p(n,n) % 1000000 != 0):
        print n
        n+=1
    print n

def problem23():
    #all numbers > 28123 can be written as a sum of two abundant numbers
    upper_bound = 28123
    #12 is the smallest abundant number
    lower_bound = 12
    import time
    from math import sqrt, ceil
    def getfactors(n):
        factors = [1]
        for i in range(2,int(ceil(sqrt(n)))+1):
            if n % i == 0:
                if i not in factors:
                    factors.append(i)
                    factors.append(n/i)
        #perfect square roots are added twice to factors, remove one.
        if sqrt(n) in factors:
            factors.remove(sqrt(n))
            
        return factors

    def is_abundant(n):
        return sum(getfactors(n)) > n

    t0 = time.time()
    abundant_list = [12]
    for i in xrange(13,upper_bound+1):
        if (is_abundant(i)):
            abundant_list.append(i)
    
    num_line = [0] * upper_bound
    for a in abundant_list:
        for b in abundant_list:
            if (a + b < upper_bound):
                num_line[a+b] += 1

    tot = 0
    for i in xrange(1,len(num_line)):
        if num_line[i] == 0:
            tot += i
    print tot
    t1 = time.time()
    dt = t1 - t0
    print dt

def problem24():
    from itertools import permutations
    vals = range(10)
    perms = list(permutations(vals))
    print perms[999999]

def problem66():
    import math

    def chakravala(n):
        """
        The chakravala method of solving indeterminate quadratic equations.

        returns the array [x,y] that solves the equation x^2 = Ny^2 + 1

        note: x will always be the minimum x for each N
        """

        if math.sqrt(n) % 1 == 0:
            #No positive solutions of x or y can be found if n is a
            #perfect square.
            
            #raise ValueError, "n cannot be square."
            return (0)

        #Make first guess of x = rounded(sqrt(n)), y = 1

        current_p = current_x = optimal_p = round(math.sqrt(n))
        current_y = 1
        
        current_k = (current_p**2) - n

        #print current_p, current_k, current_x, current_y
        
        #First guess happened to be correct, return it.
        if current_k == 1:
            return (current_x, current_y)

        #First guess was incorrect, time to modify guess.

        while True:
            #calculate how far optimal_p is off from being valid
            diff = (current_p + optimal_p) % abs(current_k)
            next_p_low = optimal_p - diff
            next_p_high = next_p_low + abs(current_k)

            #Choose next_p such that next_p**2 - n is minimized
            if next_p_low < 1:
                next_p = next_p_high
            else:
                if abs((next_p_low**2 - n)) < abs((next_p_high**2 - n)):
                    next_p = next_p_low
                else:
                    next_p = next_p_high


            #Calculate next triplet from next_p
            next_k = ((next_p**2) - n) / current_k
            next_x = ((next_p * current_x) + (n * current_y)) / abs(current_k)
            next_y = ((next_p * current_y) + current_x) / abs(current_k)

            #print next_p, next_k, next_x, next_y
            
            if next_k == 1:
                return (next_x, next_y)

            current_p = next_p
            current_k = next_k
            current_x = next_x
            current_y = next_y

    print max([(chakravala(d), d) for d in range(2,1001)])
    
def problem443():

    def bigrange(stop):
        i = 5
        while i < stop:
            yield i
            i+=1
    
    def gcd(a,b):
        """ the euclidean algorithm
            calculates the greatest common divisor of a and b
        """
        while a:
                a, b = b%a, a
        return b

    def g(n):
        if n <= 4:
            return 13
        g_n_minus_one = 13
        g_n = 0
        for i in bigrange(n+1):
            #print n, g_n, g_n_minus_one, gcd(g_n_minus_one, i)
            g_n = g_n_minus_one + gcd(g_n_minus_one,i)
            g_n_minus_one = g_n

        return g_n

    
    print g(1000000000000000)

def problem27():
    from math import ceil, sqrt
    def isprime(n):
        
        if n <= 1:
            return False
        elif n in [2,3]:
            return True
        
        upper = int(ceil(sqrt(n)))
        for i in xrange(2,upper+2):
            if n % i == 0:
                return False
        return True

    best_count = 0
    best = (0,0)
    for b in xrange(1,1001):
        if (isprime(b)):
            for a in xrange(-1000,1001):
                n = 0
                done = False
                while not done:
                    candidate = n * n + a * n + b
                    if not isprime(candidate):
                        done = True
                    else:
                        n+=1
                if n > best_count:
                    best_count = n
                    best = (a,b)
    print best

def problem28():
    """
    Finding the sum of the diagonals of a 1001x1001 spiral.
    Note: The formulae for the corners are:
    
    UR = n * n
    UL = n * n - n + 1
    LL = n * n - 2 * n + 2
    LR = n * n - 3 * n + 3
    
    """
    max_len = 1001
    tot = 1
    n = 3 #length of one side of the sub-square in the spiral
    while(n <= max_len):
        tot += n * n #upper right corner
        tot += n * n - n + 1 #upper left corner
        tot += n * n - 2 * n + 2 #lower left corner
        tot += n * n - 3 * n + 3 #lower right corner
        n+=2 #add one row and one column to sub-square
    print tot

def problem29():
    seq = []
    for a in xrange(2,101):
        for b in xrange(2,101):
            ans = a**b
            if not ans in seq:
                seq.append(ans)
    print len(seq)

def problem30():
    """
    Find the sum of all the numbers that can be written as the sum
    of the fifth powers of their digits.
    """
    def issum(n):
        tot = 0
        for letter in str(n):
            digit = int(letter)
            tot += digit * digit * digit * digit * digit #digit^5
        return n == tot

    numlist = []
    n = 2
    while( n <= 1000000):
        if issum(n):
            numlist.append(n)
        n+=1

    print numlist
    print sum(numlist)
    ### All possible numbers happened to be < 1000000

def problem31():
    """
    How many different ways can 2 pounds be made using any number of coins?
    Denominations are: 1p, 2, 5p, 10p, 20p, 50p, 1L(100p), 2L(200p)
    """
    #Exercise in Dynamic Programming to solve

    coins = [1, 2, 5, 10, 20, 50, 100, 200]
    
    #Generate a 200 x 8 matrix
    matrix = [ [0 for col in range(8)] for row in range(201)]
    
    TARGET = 200
    #there is one way to make every solution using just 1p coins, fill
    #in the matrix to reflect that.
    for n in xrange(0, TARGET+1):
        matrix[n][0] = 1

    for row in xrange(0,TARGET+1):
        for col in xrange(1, len(coins)):
            #Is the coin value of col bigger than the target value of row?
            if row >= coins[col]:
                #Target sum can be obtained via two ways:
                # 1. The number of ways to form this target
                # using only coins less than col
                matrix[row][col] += matrix[row][col-1]

                # 2. The number of ways to form this target value when using the
                # coin value of col. Subtract value of coin(col) from
                # target (row) and add the solution to the subproblem.
                matrix[row][col] += matrix[row-coins[col]][col]

            else:
                #Coin cannot be used, target is too small to allow coin.
                #No solution exists with coin then.
                matrix[row][col] = matrix[row][col-1]

    print matrix[-1][-1]

def problem32():
    def ispandigital(n):
        assert type(n) is str, 'n must be a string of digits.'
        vals = ['1','2','3','4','5','6','7','8','9']
        for letter in n:
            if not letter in vals:
                return False
            else:
                vals.remove(letter)
        if len(vals) != 0:
            return False
        return True

    def getidentity(a,b):
        out = ''
        out += str(a)
        out += str(b)
        out += str(a*b)
        return out

    productlist = []
    for a in xrange(1,2000):
        for b in xrange(1,2000):
            mystr = getidentity(a,b)
            if len(mystr) == 9:
                if ispandigital(mystr):
                    #prevent overlapping from when a * b = b *a
                    product = a * b
                    if not product in productlist:
                        productlist.append(product)
    print productlist
                
                
def problem19():
    """1 Jan 1900 was a Monday.
        April June September November - 30 days
        
        January March May July August October December - 31 days

        February: If year is divisible by 4 and is NOT on a century
            UNLESS it is divisible by 400. - 29
            else - 28

        How many sundays fell on the first of the month during the twentieth century
        (1 Jan 1901 - 31 Dec 2000)?
    """
    #Use Python's datetime module to create a date object for the first
    #of the month of each month in the year, then have each object return the
    #weekday (0 - Monday 6 - Sunday).
    
    import datetime

    count = 0
    for year in xrange(1,101):
        for month in xrange(1,13):
            day = datetime.date(year + 1900, month, 1)
            if (day.weekday() == 6):
                count += 1
    print count
    

def problem33():
    """
    Curious fractions are fractions that yield the correct answer even when
    incorrectly canceled out.

    Example: 49/98 = 4/8 (when 9s are incorrectly canceled)
    even though incorrectly cancalled, 49/98 == 4/8.

    Exclude trivial fractions of the form 30/50 == 3/5
    
    Find the four non-trivial examples of this fraction  < 1
    Find the value of the denominator of the product of the four fractions
    given in its lowest common terms.
    """
    from fractions import gcd

    #When does ax/xb == a / b
    def iscurious(a,b):
        for x in xrange(0,10):
            num = a*10 + x
            denom = x*10 + b
            if num/float(denom) == a/float(b):
                print a,b,(a*10 + x), "/", (x*10 + b)
                return True
        return False

    for a in range(1,10):
        for b in range(1,10):
            if a == b:
                pass
            elif iscurious(a,b):
                print a, b
            
    #16/64, 19/95, 49/98, 26/65

def problem34():
    """
    Find the sum of all numbers that are equal ot the sum of the factorial
    of their digits.

    Ex: 145 = 1! + 4! + 5! = 1 + 24 + 120 = 145
    """
    def factorial(n):
        prod = 1
        for x in xrange(1,n+1):
            prod *= x
        return prod

    def getdigits(n):
        digitslist = []
        for letter in str(n):
            digitslist.append(int(letter))
        return digitslist

    def iscurious(n):
        digitslist = getdigits(n)
        tot = 0
        for digit in digitslist:
            tot += factorial(digit)
            
        if tot == n:
            return True
        else:
            return False

    candidate = 3
    numlist = []
    while candidate < 100000:
        if iscurious(candidate):
            numlist.append(candidate)
        candidate += 1
    print numlist

def problem35():
    """
    Circular if is prime AND all rotations of the digits are also prime.
    """
    from math import sqrt, ceil
    from copy import copy
    
    def isprime(n):
        if n < 1:
            return False
        elif n in [2,3]:
            return True
        upper = int(ceil(sqrt(n)))
        for i in xrange(2, upper+2):
            if n % i == 0:
                return False
        return True

    def getrotations(n):
        num = []
        for letter in str(n):
            num.append(letter)
        rotations = []
        x = 0
        while x < len(num):
            num.insert(0, num.pop())
            rotations.append(copy(num))
            x+=1

        out = []
        for i in xrange(len(rotations)):
            out.append(int(''.join(rotations.pop())))

        return out

    def iscircular(rotations):
        """
        A prime is circular if all the rotations of its digits
        are also prime.
        """
        for rot in rotations:
            if not isprime(rot):
                return False
        return True

    index = 2
    circlist = []
    while(index < 1e6):
        if isprime(index):
            rotations = getrotations(index)
            if iscircular(rotations):
                circlist.append(index)
        index +=1
    print circlist

def problem36():
    """
    Find all numbers less than one million that are palindromic in both
    base 10 and base 2.

    Eg. 585 = 1001001001 (base 2)
    """

    def ispalindrome(n):
        return n == n[::-1]

    def getbinary(n):
        """
        Returns the binary form of the decimal n.
        """

        return int(bin(n)[2:])


    candidate = 0
    numlist = []
    while candidate < 1e6:
        if ispalindrome(str(candidate)):
            binary = str(getbinary(candidate))
            if ispalindrome(binary):
                numlist.append(candidate)
        candidate += 1
    print numlist
            

def problem37():
    """
    Trunctable primes. Primes that are also prime for truncates in both
    a left-to-right and a right-to-left format.

    There are only eleven primes that are also truntable, find them.
    """
    from math import ceil, sqrt
    def isprime(n):
        if n == 1:
            return False
        elif n in [2,3]:
            return True

        upper = int(ceil(sqrt(n)))
        for i in xrange(2, upper + 2):
            if n % i == 0:
                return False
        return True

    
    def truncate(n):
        """
        Removes digits from left to right until single digit remains.

        Then removes digits from right to left until a single digit remains.
        """
        #left-to-right
        numlist = []
        mystr = str(n)
        index = 0
        while index < len(mystr):
            numlist.append(int(mystr[index:]))
            index += 1

        #right-to-left
        index = len(mystr)
        while index > 0:
            numlist.append(int(mystr[:index]))
            index -= 1
        return numlist
    
    UPPER_LIMIT = 1e6
    candidate = 8
    results = []
    while candidate < UPPER_LIMIT:
        if isprime(candidate):
            parts = truncate(candidate)
            success = True
            for part in parts:
                if not isprime(part):
                    success = False
            if success:
                results.append(candidate)
        candidate += 1
    print results

def problem38():
    """
    What is the largest 1 to 9 pandigital 9-digit number that can be formed
    as the concatenated product of an integer with (1,2,...,n), where n > 1?
    """
    def ispandigital(n):
        assert type(n) is str, 'n must be a string of digits.'
        vals = ['1','2','3','4','5','6','7','8','9']
        for letter in n:
            if not letter in vals:
                return False
            else:
                vals.remove(letter)
        if len(vals) != 0:
            return False
        return True

    for n in xrange(9876, 9123, -1):
        candidate = str(n * 1) + str(n * 2)
        if ispandigital(candidate):
            print candidate
            break
    
    
def problem39():
    """
    For which value of p <= 1000 is the number of solutions maximized?
    """
    from fractions import gcd
    from math import ceil, sqrt

    #As c = m * m and p = a + b + c, m must be upper bound by at most
    #the square root of 1000
    upper_bound = int(ceil(sqrt(1000)))
    def evenodd(m,n):
        """ Test to see if at least one of (m,n) is even and
            one of (m,n) is odd.
        """
        if (m + n) % 2 == 0:
            return False
        return True

    solutions = [0] * 1001
    for m in xrange(1, upper_bound + 1):
        for n in xrange(1,m):
            if evenodd(m,n) and gcd(m,n) == 1:
                a = m*m - n*n
                b = 2*m*n
                c = m*m + n*n
                p = a + b + c
                while p <= 1000:
                    solutions[p] += 1
                    p += (a + b + c)

    print solutions.index(max(solutions))    

def problem41():
    """
    What is the largest n-digit pandigital prime that exists for 1-n digits?
    """
    from math import ceil, sqrt

    def isprime(n):
        if n == 1:
            return False
        elif n in [2,3]:
            return True

        upper = int(ceil(sqrt(n)))
        for i in xrange(2, upper + 2):
            if n % i == 0:
                return False
        return True
    
    def ispandigital(n):
        assert type(n) is str, 'n must be a string of digits.'
        vals = []
        for i in xrange(1, len(n)+1):
            vals.append(str(i))
        for letter in n:
            if not letter in vals:
                return False
            else:
                vals.remove(letter)
        if len(vals) != 0:
            return False
        return True

    best = 0
    for n in xrange(1000000,10000000):
        if isprime(n):
            candidate = str(n)
            if ispandigital(candidate):
                if candidate > best:
                    best = candidate
    print best
    print 'Finished.'

    #Best so far: 7652413
    #Note: no solution for 9 or 8 digit numbers.
    #(1+2+3+4+5+6+7+8+9) = 45 => always divisible by 3
    #(1+2+3+4+5+6+7+8) = 36 => always divisible by 3

def problem42():
    #create Dictionary that maps character to index in alphabet
    keys = [
        'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
        'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
        'W', 'X', 'Y', 'Z'
        ]
    values = range(1,27)
    alphadict = dict(zip(keys, values))

    #Create list of words from file
    infile = open('words.txt', 'r')
    data = infile.read().split(",")
    wordlist = []
    for word in data:
        wordlist.append(word.strip('"'))

    #Create Triangle sequence: tn = 1/2*n*(n+1)
    #Longest word: 'Administration' = 14 characters
    upper_bound = 364
    triangles = []
    n = 1
    while n < upper_bound:
        triangles.append(int(0.5*n*(n+1)))
        n+=1

    count = 0
    for word in wordlist:
        score = 0
        for letter in word:
            score += alphadict[letter]
        if score in triangles:
            count += 1
    print count

def problem44():
    """
    Find the pair of pentagonal numbers Pj and Pk, for which their sum and
    difference are pentagonal and D = |Pj - Pk| is minimised.
    """
    from math import sqrt
    def ispentagonal(n):
        test = (sqrt(1+24*n) + 1) / 6.0
        return test == int(test)

    found = False
    result, i = 0, 1

    while not found:
        i+=1
        n = i * (3 * i - 1) / 2
        for j in xrange(i-1, 0, -1):
            m = j * (3 * j - 1) / 2
            if ispentagonal(n - m) and ispentagonal(n + m):
                result = n - m
                found = True
                break
    print i, j, result

def problem45():
    """
    It is known that the 285th Triangular number happens to be the 165th
    Pentagonal number and the 143rd Hexagonal number. Find the next one.
    """
    from math import sqrt
        
    def ispentagonal(n):
        test = (sqrt(1+24*n) + 1) / 6.0
        return test == int(test)

    def ishexagonal(n):
        test = (sqrt(8*n+1) + 1) / 4.0
        return test == int(test)

    i = 286
    success = False
    while not success:
        n = i * (i + 1) / 2
        if ispentagonal(n):
            if ishexagonal(n):
                print n
                success = True
        i += 1

def problem46():
    """
    Disprove Goldbach's other conjecture that every odd composite number can be
    written as the sum of a prime and a square.
    """
    from math import sqrt

    def is_prime(n):
        if n == 1:
            return False
        elif n in [2,3]:
            return True

        upper = int(sqrt(n)) + 2
        for i in xrange(2, upper + 2):
            if n % i == 0:
                return False
        return True

    def is_square(n):
        return sqrt(n).is_integer()
    
    def is_odd_composite(n):
        if n % 2 == 0:
            return False

        upper = int(sqrt(n)) + 2
        for i in xrange(2, upper):
            if n % i == 0:
                return True
        return False

    def fits_constraint(n):
        primelist = []
        for i in xrange(2,n):
            if is_prime(i):
                primelist.append(i)

        for prime in primelist:
            candidate = (n - prime) / 2.0
            if is_square(candidate):
                return True
        return False

    n = 34
    while True:
        if is_odd_composite(n):
            if not fits_constraint(n):
                print n
                break
        n += 1

    #5777 cannot be made by a prime plus 2 * a square number

def problem50():
    def sieve(limit):
        marked = [0] * limit

        marked[0] = marked[1] = 1

        for x in xrange(2, limit, 2): #Even numbers cannot be prime
            marked[x] = 1

        for x in xrange(3, limit, 2):
            if marked[x] == 0:
                for y in xrange(x*2, limit, x):
                    marked[y] = 1

        return marked

    primebits = sieve(int(1e6))
    primes = []
    primesums = []

    for x in xrange(int(1e6)):
        if primebits[x] == 0:
            primes.append(x)
    limit = len(primes)/2
    maximum = 0

    for x in xrange(limit):
        tot = 0
        count = 0
        for y in xrange(x, limit):
            tot += primes[y]
            count += 1
            if tot > 999999: break
            if primebits[tot] == 0:
                primesums.append((tot, count))
                if count > maximum: maximum = count
        if limit - x < maximum: break
        #print x

    print 'out'

    for (x,y) in primesums:
        if y == maximum: print x

def problem52():
    """
    The number 125874 and its double, 251748, contain exactly the same digits,
    but in different order.

    Find the smallest positive integer, x, such that 2x, 3x, 4x, 5x, and 6x,
    contain the same digits.

    NOTE: Updated with is_perm, much more efficient way to check
    if they are permutations of each other.
    """
    #from itertools import permutations
    import time
    def is_perm(str1, str2):
        list1 = [x for x in str1]
        list2 = [x for x in str2]
        return set(list1) == set(list2)
    
    def fitsconstraint(n):
        assert type(n) is str, 'n must be a string'
        tests = [2,3,4,5,6]
##        perms = list(permutations(n))
##        permset = []
##        for perm in perms:
##            permset.append(''.join(perm))

        for test in tests:
            newnum = str(int(n) * test)
            if not is_perm(n, newnum):
                return False
        return True

    def sieve(n):
        upper = str(6*n)
        return len(str(n)) == len(upper)

    def main():
        t0 = time.time()
        n = 1
        while True:
            if sieve(n):
                candidate = str(n)
                if fitsconstraint(candidate):
                    print candidate
                    break
            n += 1
        t1 = time.time()
        print (t1-t0)
        
    if __name__ == "__main__":
        main()

def problem53():
    """
    Count the number of combations of 1<= n <= 100 that are greater than
    one million.
    """
    def fact(n):
        prod = 1
        i = 1
        while i <= n:
            prod *= i
            i+=1
        return prod
    
    def C(n,r):
        num = fact(n)
        denom = fact(r) * fact(n-r)
        return num/denom
    
    def main():
        count = 0
        for n in xrange(1,101):
            for r in xrange(0,n+1):
                if C(n,r) > 1000000:
                    count += 1
        print count
                
    if __name__ == "__main__":
        main()

def problem56():
    """
    Considering natural numbers of hte form a^b, where a,b < 100, what is the
    maximum digital sum?
    """

    def sumofdigits(n):
        tot = 0
        for digit in str(n):
            tot += int(digit)
        return tot

    def main():
        besta, bestb, bestsum = 0,0,0
        for a in xrange(1,100):
            for b in xrange(1,100):
                currentsum = sumofdigits(pow(a,b))
                if currentsum > bestsum:
                    besta, bestb, bestsum = a,b,currentsum
        print besta, bestb, bestsum

    if __name__ == "__main__":
        main()

def problem57():
    """
    In the first one-thousand expansions of the infinite continued fraction
    expansion of the square root of two, how many fractions contain a numerator with more
    digits than the denominator?
    """

    #sqrt(2) can be expressed as a continued fraction of Pell numbers P(n).
    #solutions have the form: P(n-1)+P(n)/P(n) for the nth expansion.
    memo = {}
    def P(n):
        if not n in memo:
            if n < 0:
                return 0
            if n == 1:
                return 1
            else:
                memo[n] = (2*P(n-1)) + P(n-2)
        return memo[n]
        
    def main():
        count = 0
        for n in xrange(1, 1001):
            num = str(P(n-1) + P(n))
            denom = str(P(n))
            if len(num) > len(denom):
                count += 1
        print count
            
    if __name__ == "__main__":
        main()

def problem58():
    """
    For what side length of ulam's spiral does the ratio of primes along both
    diagonals drop below 10%?
    NOTE:
    For a sub-spiral of length n, the corners are as follows:
    UR = n * n
    UL = n * n - n + 1
    LL = n * n - 2 * n + 2
    LR = n * n - 3 * n + 3
    """
    from math import ceil, sqrt

    def isprime(n):
        if n == 1:
            return False
        elif n in [2,3]:
            return True

        upper = int(ceil(sqrt(n)))
        for i in xrange(2, upper + 2):
            if n % i == 0:
                return False
        return True
    
    def getcorners(n):
        """
        Returns a list of the corners of ulam's sub-spiral of lenght n.
        """
        corners = []
        corners.append(n * n - 3 * n + 3)
        corners.append(n * n -2 * n + 2)
        corners.append(n * n - n + 1)
        corners.append(n * n)
        return corners

    def countprimes(diagonals):
        count = 0
        for num in diagonals:
            if isprime(num):
                count += 1
        return count

    def main():
        primecount, totalcount = 3, 5
        n = 5
        while (primecount / float(totalcount) * 100) >= 10.0:
            primecount += countprimes(getcorners(n))
            totalcount += 4
            n += 2
        print n

    if __name__ == "__main__":
        main()

def problem59():
    """
    Using the file containing the encrypted ASCII codes and the knowledge
    that the plain text must contain common English words, decrypt the message
    and find the sum of ASCII values in the original text.
    Note: Encryption key consists of three lower case characters for a possible
          range of 97-122.

        ord() returns the ascii value of a character
        chr() returns the character of an ascii value
    """
    import time
    f = open('cipher1.txt', 'r')
    data = map(lambda x: int(x), f.read().split(','))
    words = ['in', 'or', 'the', 'and', 'but']

    def decrypt(password):
        msg, i = [], 0
        for letter in data:
            msg.append(chr(letter ^ ord(password[i % 3])))
            i += 1
            
        out = ''.join(msg)
        return out

    def countmatches(msg):
        matches = 0
        for word in words:
            if word in msg:
                matches += 1
        return matches

    def main():
        t0 = time.time()
        best, bestmsg, bestpass = 0, '', ''
        lower, upper = ord('a'), ord('z') + 1
        for a in xrange(lower, upper):
            for b in xrange(lower, upper):
                for c in xrange(lower, upper):
                    guess = '' + chr(a) + chr(b) + chr(c)
                    message = decrypt(guess)
                    matches = countmatches(message)
                    if matches > best:
                        best, bestmsg, bestpass = matches, message, guess
        print bestpass
        t1 = time.time()
        print t1 - t0

    def main2():
        print decrypt('god')
        
    if __name__ == "__main__":
        main()

def problem60():
    """
    Find the lowest sum for a set of five primes for which any two primes
    concatenate to produce another prime.
    """
    from math import sqrt
    
    def isprime(n):
        if n == 1:
            return False
        elif n in [2,3]:
            return True

        upper = int(sqrt(n)) + 2
        for i in xrange(2, upper + 2):
            if n % i == 0:
                return False
        return True
    
    def concatenate(a,b):
        return int(str(a) + str(b))

    def fitsconstraint(n,primes):
        for prime in primes:
            candidate1 = concatenate(prime, n)
            if not isprime(candidate1):
                return False
            candidate2 = concatenate(n, prime)
            if not isprime(candidate2):
                return False
        return True
            

    def main():
        primes = [3,7,109]
        i = 670
        done = False
        while not done:
            if isprime(i):
                if fitsconstraint(i,primes):
                    primes.append(i)
                    done = True
            i+=1
        print primes
        
    if __name__ == "__main__":
        main()

def problem62():
    """
    Find the smallest cube for which exactly five permutations of its digits are
    cube.
    """
    from time import clock
    def sortdigits(n):
        return ''.join(sorted([digit for digit in str(n)]))

    def main():
        t0 = clock()
        occurCount,solution,i = {},-1,1
        while True:
            cube = i**3
            digitTuple = tuple(sortdigits(cube))
            if not digitTuple in occurCount:
                occurCount[digitTuple] = [0,cube]
            occurCount[digitTuple][0] += 1
            if occurCount[digitTuple][0] == 5:
                solution = occurCount[digitTuple][1]
                break
            i+=1
        print solution
        print clock() - t0
        
    if __name__ == "__main__":
        main()

def problem63():
    """
    How many n-digit positive integers exist which are also an nth power?
    """
    import time
    def fitsconstraint(k,n):
        return len(str(pow(k,n))) == n

    def main():
        t0 = time.time()
        count = 0
        #assume range will be in 1^1 through 100^100
        for k in xrange(1,100):
            for n in xrange(1,100):
                if fitsconstraint(k,n):
                    count += 1
        t1 = time.time()
        print count
        print t1 - t0

    if __name__ == "__main__":
        main()

def problem65():
    """
    Find the sum of digits in the numerator of the 100th convergent of the
    continued fraction for e.
    """
    from time import clock

    def a(n):
        if n == 0:
            return 2
        else:
            i = n % 3
            if i == 2:
                return  ((n+1)/3) * 2
            return 1
        
    #numerator(n) = a(n) * h(n-1) + h(n-2)
    #http://en.wikipedia.org/wiki/Continued_fraction#Some_useful_theorems
    def getnumerator(n):
        n_minus_two = 0
        n_minus_one = 1
        ans,i = 0,0
        while i < n:
            ans = a(i) * n_minus_one + n_minus_two
            n_minus_two = n_minus_one
            n_minus_one = ans
            i+=1
        return ans

    def getdenom(n):
        k_minus_two = 1
        k_minus_one = 0
        ans, i = 0,0
        while i < n:
            ans = a(i) * k_minus_one + k_minus_two
            k_minus_two = k_minus_one
            k_minus_one = ans
            i+=1
        return ans
        
    def main():
        t0 = clock()
        num = getnumerator(100)
        print sum([int(digit) for digit in str(num)])
        print clock() - t0

    if __name__ == "__main__":
        main()

def problem68():
    """
    Using the numbers 1 to 10, and depending on the arrangements, its possible
    to form  16- and 17- digit strings. What is the maximum 16-digit string
    for the 'magic' 5-gon ring?
    """
    #Rather than use networkx to create a graph, use a 1d array.
    #With "lines" as follows: 0-1-2, 3-2-4, 5-4-6, 7-6-8, 9-8-1
    #Outer vertices: 0, 3, 5, 7, 9
    from itertools import permutations
    from time import clock

    def is_valid(arr):
        vectors = (0,1,2), (3,2,4), (5,4,6), (7,6,8), (9,8,1)
        solutions = set() 
        for i,j,k in vectors:
            solutions.add(arr[i] + arr[j] + arr[k])
        return len(solutions) == 1

    def getset(arr):
        vectors = (0,1,2), (3,2,4), (5,4,6), (7,6,8), (9,8,1)
        mval, midx = min((val, idx) for (idx, val) in enumerate(
                            [arr[0], arr[3], arr[5], arr[7], arr[9]]
                            ))
        shifted = vectors[midx:] + vectors[:midx]
        this_set =[]
        for i,j,k in shifted:
            this_set.append(arr[i])
            this_set.append(arr[j])
            this_set.append(arr[k])
        return ''.join(str(num) for num in this_set)
        
    
    def main():
        t0 = clock()
        vals = range(1,11)
        perms = list(permutations(vals))
        solution_set = set()
        for perm in perms:
            if is_valid(perm):
                ans = getset(perm)
                if len(ans) == 16:
                    solution_set.add(int(ans))

        print max(solution_set)            

    def test():
        def is_valid(arr):
            vectors = (0,1,2), (3,2,4), (5,4,1)
            solutions = set()
            for i,j,k in vectors:
                 solutions.add(arr[i] + arr[j] + arr[k])
            return len(solutions) == 1

        myarray = [4,3,2,6,1,5]
        
        print is_valid(myarray)
            
    if __name__ == "__main__":
        main()

def problem69():
    """
    Find the value of n <= 1,000,000 for which n/phi(n) is a maximum.
    """
    from time import clock
    from fractions import gcd
    #Note: gcd algorithm is euclidean, not binary
    
    def sieve(n):
        size = n//2
        sieve = [1] * size
        limit = int(n**0.5)
        for i in xrange(1,limit):
            if sieve[i]:
                val = 2*i+1
                tmp = ((size-1) - i)//val
                sieve[i+val::val] = [0]*tmp
        return [2] + [i*2+1 for i, v in enumerate(sieve) if v and i > 0]

    def phi(n, primes):
        if n in primes:
            return n-1
        else:
            count = 1
            for i in xrange(2,n):
                if gcd(n,i) == 1:
                    count +=1
            return count
        
    def main():
        UPPER = int(1e6)
##        primes = sieve(UPPER)
##        bestnum,bestval = 0,0
##        for i in xrange(2,UPPER+1):
##            val = i/phi(i,primes)
##            if val > bestval:
##                bestnum, bestval = i, val
##                
##        print bestnum, bestval
        primes = sieve(UPPER)
        current, i, t0 = 1, 0, clock()
        while(current * primes[i] < UPPER):
            current *= primes[i]
            i+=1
        print current
        print clock() - t0
        
    if __name__ == "__main__":
        main()

def problem70():
    """
    Find the value of n, 1 < n < 10^7, for which phi(n) is a permutation of n
    and the ratio n/phi(n) produces a minimum.
    """
    from time import clock
    
    def sieve(n):
        size = n//2
        sieve = [1] * size
        limit = int(n**0.5)
        for i in xrange(1,limit):
            if sieve[i]:
                val = 2*i+1
                tmp = ((size-1) - i)//val
                sieve[i+val::val] = [0]*tmp
        return [2] + [i*2+1 for i, v in enumerate(sieve) if v and i > 0]

    def sortdigits(n):
        return ''.join(sorted([digit for digit in str(n)]))

    def isperm(a,b):
        return sortdigits(a) == sortdigits(b)
    
    def main():
        UPPER = int(1e7)
        primes = sieve(10000)
        i,j = 0,0
        while primes[i] < 2000:
            i+=1
            j+=1
        while primes[j] <= 5000:
            j+=1
        primes = primes[i:j]
        
        t0 = clock()
        bestn, bestratio = 2,9
        i = 0
        for p1 in primes:
            i+=1
            for p2 in primes[i:]:
                n = p1*p2
                if n <= 10**7:
                    phi = (p1-1) * (p2-1)
                    if isperm(n,phi):
                        ratio = n/float(phi)
                        if ratio < bestratio:
                            bestn, bestratio = n, ratio
        print bestn, bestratio
        print clock() - t0
                    
                
    if __name__ == "__main__":
        main()

def problem71():
    """
    A fraction is a reducd proper fraction if n/d are positive integers,
    n < d and gcd(n,d) = 1

    By listing the set of reduced proper fractions for d <= 1,000,000 in
    ascending order of size, find the nuerator of the fraction immediately
    to the left of 3/7.
    """
    
def problem72():
    """
    A fraction is a reduced proper fraction if n/d are positive integers,
    n < d, and gcd(n,d) = 1

    How many fractions for d <= 1,000,000 are reduced proper fractions?
    """
    #Trying to do this with gcd would take too much time.
    #Note: gcd(n,d) == 1 will occur if n and d are relatively prime
    #Therefore problem can be restated as the sum of the phi(i)
    #for 2 <= i <= 1e6.

    from operator import mul #to find product of an array
    from time import clock
    from math import sqrt, ceil

    def factor(n):
        if n <= 1: return
        prime = next((x for x in range(2, int(ceil(sqrt(n)))+1) if n%x == 0), n)
        yield prime
        for p in factor(n//prime):
            yield p
    
    def sieve(n):
        size = n//2
        sieve = [1] * size
        limit = int(n**0.5)
        for i in xrange(1,limit):
            if sieve[i]:
                val = 2*i+1
                tmp = ((size-1) - i)//val
                sieve[i+val::val] = [0]*tmp
        return [2] + [i*2+1 for i, v in enumerate(sieve) if v and i > 0]

##    UPPER = int(1e6)
##    primes = sieve(UPPER)
    
    def yieldprimes(n):
        i = 0
        while primes[i] < int(sqrt(n)+2):
            yield primes[i]
            i+=1
            
    def phi(n):
        if n == None:
            return float(n-1)
        else:
            total = n
            for v in set(factor(n)):
                    total *= (1 - (1.0/v))
            return total
    
    def main():
        t0 = clock()
        print int(sum(phi(n) for n in xrange(2, 1000001)))
        print clock() - t0
            
    if __name__ == "__main__":
        main()
    #303963552391
        
def problem74():
    """
    How many chains, with a starting number below one million, contains
    exactly sixty non-repeating terms?
    """
    from time import clock
    #Factorial sums for digits 0-9
    fsums = [1,1,2,6,24,120,720,5040,40320,362880]

    def sumoffacts(n):
        total = 0
        for digit in str(n):
            total += fsums[int(digit)]
        return total

    def getchainlength(n):
        count = 0
        seq = []
        while not n in seq:
            seq.append(n)
            n = sumoffacts(n)
            count += 1
        return count
    
    def main():
        #cheating time, if any number is in signatures, it will have a
        # chain of 60.
        signatures = [367945, 367954, 373944, 379443, 379465, 735964]
        count = 0
        t0 = clock()
        for i in xrange(1, int(1e6)):
            if sumoffacts(i) in signatures:
                count += 1
        print count
        print clock() - t0

    if __name__ == "__main__":
        main()
                
def problem76():
    """
    How many different ways can one hundred be written as a sum of at least
    two positive integers?
    """
    from time import clock
    partitions = {}
    def p(n):
        if n < 0: return 0
        if n == 0: return 1
        if n not in partitions:
            partitions[n] = sum([(-1)**(k+1) * (p(n - (k * (3 * k - 1)/2))
                + p(n - (k * (3 * k + 1) / 2))) for k in xrange(1, n+1)])
        return partitions[n]

    def main():
        t0 = clock()
        print p(100) - 1
        print clock() - t0

    if __name__ == "__main__":
        main()

def problem79():
    """
    Analyse the file so as to determine the shortest possible secret passcode
    of unknown length using the file keylog.txt
    """
    
    fname = 'keylog.txt'
    f = open(fname, 'r')
    data = []
    for line in f.readlines():
        data.append(line.strip('\n'))

    freqdict = {}
    for number in data:
        index = 1
        for digit in number:
            data = (digit, index)
            if not data in freqdict:
                freqdict[data] = 0
            freqdict[data] += 1
            index += 1
    print freqdict

    #73162890, worked it out by hand from frequencies.

def problem81():
    """
    Find the minimal path sum in the 80x80 matrix given in matrix.txt moving
    from the top left to the bottom right by only moving right and down.
    """
    from time import clock
    MAX_ROW = MAX_COL = 79
    
    def loadmatrix():
        fname = 'matrix.txt'
        f = open(fname, 'r')
        matrix = []
        for row in f:
            matrix.append([int(n) for n in row.split(',')])
        print "Matrix loaded."
        return matrix

    def get_shortest_path(matrix):
        for i in xrange(78, -1, -1):
            matrix[79][i] += matrix[79][i+1]
            matrix[i][79] += matrix[i+1][79]

        for i in xrange(78, -1, -1):
            for j in xrange(78, -1, -1):
                matrix[i][j] += min(matrix[i+1][j], matrix[i][j+1])
            
        print matrix[0][0]            

    def main():
        matrix = loadmatrix()
        t0 = clock()
        get_shortest_path(matrix)
        print clock() - t0
        
    if __name__ == "__main__":
        main()

def problem81_a_star():
    """
    Same premise as problem 81, but solved with the A* algorithm.
    """
    from math import sqrt
    import heapq
    
    class node():
        def __init__(self, row, col, hscore, val=None, parent=None):
            self.row = row
            self.col = col
            if val is None:
                self.val = 9999
            else:
                self.val = val
            self.neighbors = []
            self.hscore = hscore
            self.fscore = val + self.hscore
            self.parent = parent
            
        def get_val(self):
            return self.val
        
        def get_pos(self):
            return (self.row, self.col)

        def get_h_score(self):
            return self.hscore

        def get_f_score(self):
            return self.fscore

        def update_neighbors(self, neighbors):
            for neighbor in neighbors:
                self.neighbors.append(neighbor)

        def get_neighbors(self):
            return self.neighbors

    def get_h_score(xCurr, yCurr, xDest, yDest):
        dx = xDest - xCurr
        dy = yDest - yCurr
        # Euclidean Distance
        # dist = sqrt( dx * dx + dy * dy)
        # Manhattan distance
        dist = abs(dx) + abs(dy)
        # Chebyshev distance
        # dist = max(abs(dx), abs(dy))
        return 18 * dist
            
    def loadmatrix(fname):
        f = open(fname, 'r')
        matrix = []
        for row in f:
            matrix.append([int(n) for n in row.split(',')])
        print "Matrix loaded."
        return matrix

    def a_star(start, end, matrix):
        closedset = []
        openset = []
        heapq.heapify(openset)
        heapq.heappush(openset, start)
        came_from = []
        came_from.append(start)
        lowest_sum = 0

        while openset:
            #openset = sorted(openset, key=lambda x:x.get_f_score())
            #current = openset.pop(0)
            current = heapq.heappop(openset)
            #print current.get_pos()
            #lowest_sum += current.get_val()
            if current == end:
                #print lowest_sum
                return retrace_path(current)

            closedset.append(current)
            for neighbor in current.get_neighbors():
                if neighbor not in closedset:
                    if neighbor not in openset:
                        heapq.heappush(openset, neighbor)
                        #openset.append(neighbor)
                        neighbor.parent = current
                        openset.sort(compare)
##                this_g = current.get_val() + neighbor.get_val()
##                this_f = this_g + neighbor.get_h_score()
##                if neighbor in closedset and this_f >= neighbor.get_f_score():
##                    continue
##
##                if neighbor not in openset or this_f < neighbor.get_f_score():
##                    came_from.append(neighbor)
##                    if neighbor not in openset:
##                        openset.append(neighbor)
##                    neighbor.parent = current
                        
        return "Path not found."
                
    def compare(a,b):
        if a.get_f_score() < b.get_f_score(): return -1
        elif a.get_f_score() == b.get_f_score(): return 0
        else: return 1
        
    def find_neighbors(row, col, nodes):
        #Note: Current values are for test 5x5 matrix, not the 80x80 one.
        neighbors = []
        if not row-1 < 0:
            neighbors.append(nodes[row-1][col])
        if not row+1 > len(nodes)-1:
            neighbors.append(nodes[row+1][col])
        if not col-1 < 0:
            neighbors.append(nodes[row][col-1])
        if not col + 1 > len(nodes)-1:
            neighbors.append(nodes[row][col+1])
        return neighbors

    def retrace_path(node):
        path = []
        path.append(node)
        while node.parent:
            node = node.parent
            path.append(node)
        return [node.get_pos() for node in path][::-1]
        
    def main():
        xgoal, ygoal = 79, 79
        matrix = loadmatrix('matrix.txt')
        nodes = [[None for _ in xrange(len(matrix)-1)] for _ in xrange(len(matrix)-1)]
        #First pass instantiates the nodes of the matrix
        for row in xrange(len(matrix)-1):
            for col in xrange(len(matrix)-1):
                nodes[row][col] = node(
                                row, col,
                                #find_neighbors(row, col, nodes),
                                get_h_score(row, col, xgoal, ygoal),
                                matrix[row][col]
                                       )
        #Second pass finds neighbors of each node in nodes
        for row in xrange(len(matrix)-1):
            for col in xrange(len(matrix)-1):
                nodes[row][col].update_neighbors(find_neighbors(row,col,nodes))
        
        print a_star(nodes[0][0], nodes[78][78], nodes)

    def test():
        #Testing the 5x5 matrix. Sum is 2297
        #Path is (0,0), (1,0), (1,1), (1,2), (0,2), (0,3), (0,4), (1,4)
        #(2,4), (2,3), (3,3), (4,3), (4,4)
        matrix = loadmatrix('matrix2.txt')
        #Instaniate the nodes using the values of the matrix
        xgoal, ygoal = 4, 4
        nodes = [[None for _ in xrange(len(matrix))] for _ in xrange(len(matrix))]
        for row in xrange(len(matrix)):
            for col in xrange(len(matrix)):
                nodes[row][col] = node(
                    row, col,
                    get_h_score(row, col, xgoal, ygoal),
                    matrix[row][col]
                    )
        #Second pass to find neighbors of each node.
        for row in xrange(len(matrix)):
            for col in xrange(len(matrix)):
                 nodes[row][col].update_neighbors(find_neighbors(row, col, nodes))

        print a_star(nodes[0][0], nodes[4][4], nodes)

        #Algorithm finds the correct path, need to keep track of parent nodes
        # in order to return the correct value.

    def testmatrix(filename):
        matrix = loadmatrix(filename)
        xgoal = ygoal = len(matrix)-1
        nodes = []
        nodes = [[None for _ in xrange(len(matrix)-1)] for _ in xrange(len(matrix)-1)]
        #First pass instantiates the nodes of the matrix
        for row in xrange(len(matrix)-1):
            for col in xrange(len(matrix)-1):
                nodes[row][col] = node(
                                row, col,
                                #find_neighbors(row, col, nodes),
                                get_h_score(row, col, xgoal, ygoal),
                                matrix[row][col]
                                       )
                
        for row in xrange(len(matrix)-1):
            for col in xrange(len(matrix)-1):
                if not matrix[row][col] == nodes[row][col].get_val():
                    print "Mismatch detected at: ", row, col
                #print matrix[row][col], "   ", nodes[row][col].get_val()

    if __name__ == "__main__":
        main()

#problem81_a_star()

def problem82():
    """
    Same premise as problem 81, but now starting from any location in
    column 0 and moving only up, down, and right, finishing on the rightmost
    column. Find the minimum sum.
    """
    #using networkx
    import networkx as nx
    print "NetworkX loaded."
    
    def main():
        G = nx.grid_2d_graph(80,80)
        import itertools
        from time import clock
        coordinates = list(itertools.product(xrange(80), xrange(80)))
        f = open('matrix.txt')
        data2d = [[int(i) for i in j.split(',')] for j in f]
        data1d = list(itertools.chain.from_iterable(data2d))
        valdict = dict(zip(coordinates, data1d))
        
        for node in nx.nodes_iter(G):
            G.node[node]['weight'] = valdict[node]

        weightslist = []
        for n1,n2 in nx.edges_iter(G):
            weightslist.append(tuple((n1,n2,
                                    (G.node[n1]['weight']+G.node[n2]['weight'])
                                    )))
        G.add_weighted_edges_from(weightslist)
        
        def dist(a,b):
            (x1, y1) = a
            (x2, y2) = b
            #Euclidean distance.
            #return ((x1-x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            #Manhattan distance.
            return abs((x1-x2)) + abs((y1-y2)) 
        
        t0 = clock()
        path = nx.astar_path(G,(0,0),(79,79),dist)
        print sum([G.node[num]['weight'] for num in path])
        print clock() - t0
        
    if __name__ == "__main__":
        main()

def problem83():
    """
    Same premise as problem 81, but you are now able to move up, down, left, or
    right in the matrix as you travel from (0,0) to (79,79).
    """
    import heapq
    from time import clock        

    #Dijkstra's search algorithm.
    def search(matrix):
        n = len(matrix)
        vector = (1,0), (0,1), (-1,0), (0,-1)
        visited = [[False] * n for j in xrange(n)]
        visited[0][0] = True

        l = []
        heapq.heappush(l, [matrix[0][0],0,0])
        while l:
            s, x, y = heapq.heappop(l)
            for i,j in vector:
                u = x + i
                v = y + j
                if 0 <=  u < n and 0 <= v < n:
                    if not visited[u][v]:
                        sum1 = s + matrix[u][v]
                        heapq.heappush(l, [sum1, u, v])
                        visited[u][v] = True
                    if u == n-1 and v == n - 1:
                        return s + matrix[u][v]
    def main():
        f = open('matrix.txt')
        matrix = [[int(i) for i in j.split(',')] for j in f]
        t0 = clock()
        print search(matrix)
        print clock() - t0
        
    if __name__ == "__main__":
        main()

def problem85():
    """
    By counting carefully it can be seen that a rectangular grid measuring 3 by 2
    contians eighteen triangles. Althrough there exists no rectangular grid that
    contains exactly two million rectangles, find the area of the nearest soln.
    """
    from math import factorial
    from time import clock

    #Mathematically, for an m x n matrix there will be (m+1) vertical lines and
    # (n+1) horizontal lines. Any sub-square or rectangle will be made up by
    # two of each vertical and horizaontal lines. Therefore there will be
    # ((m+1) C 2) * ((n+1) C 2) sub squares or rectangles.

    def numSubs(m, n):
        return (C(m+1, 2) * C(n+1, 2))
        
    def C(n,k):
        return (factorial(n) / (factorial (k) * (factorial (n - k))))

    def main():
        t0 = clock()
        results = {}
        for m in xrange(1,100):
            for n in xrange(m, 100):
                results[(m,n)] = abs(numSubs(m,n) - 2000000)
                
        best = min(results, key=results.get)
        print best, (best[0] * best[1])
        print clock() - t0
        
    if __name__ == "__main__":
        main()

def problem87():
    """
    How many numbers below fifty million can be expressed as the sum of a prime
    square, prime cube, and prime fourth power?
    """
    import itertools
    from math import sqrt
    from time import clock

    def powers(a,b,c):
        return a**2 + b**3 + c**4
    
    def sieve_for_primes_to(n):
        size = n//2
        sieve = [1]*size
        limit = int(n**0.5)
        for i in range(1,limit):
            if sieve[i]:
                val = 2*i+1
                tmp = ((size-1) - i)//val 
                sieve[i+val::val] = [0]*tmp
        return [2] + [i*2+1 for i, v in enumerate(sieve) if v and i>0]

    def main():
        UPPER = 50000000
        UPPER_SQRT = int(sqrt(UPPER)+1)
        primes = sieve_for_primes_to(UPPER_SQRT)
        nums = set()
        t0 = clock()
        for a in primes:
            for b in primes:
                for c in primes:
                    ans = powers(a,b,c)
                    if (ans < UPPER):
                        nums.add(ans)
                    else:
                        break
        print len(nums)
        print clock() - t0
                    
    if __name__ == "__main__":
        main()
    
def problem89():
    """
    The 11k txt file 'roman.txt' contains 11 thosuand numbers in not
    necessarily minimal, but valid roman numeral order. Find the number of
    characters saved by writing each othese in their minimum form.
    """
    cryptoDict = {'M': 1000, 'D': 500, 'C': 100, 'L': 50, 'X': 10,
                      'V': 5, 'I': 1}

    def roman_to_int(inArr):
        return sum([cryptoDict[x] for x in inArr])

    def int_to_minimal_roman(n):
        if (n / 1000 > 0):
            return "M" * (n/1000) + int_to_minimal_roman(n%1000)
        if (n / 500 > 0):
            return "D" * (n/500) + int_to_minimal_roman(n%500)
        if (n / 100 > 0):
            return "C" * (n/100) + int_to_minimal_roman(n%100)
        if (n / 50 > 0):
            return "L" * (n/50) + int_to_minimal_roman(n%50)
        if (n / 10 > 0):
            return "X" * (n/10) + int_to_minimal_roman(n%10)
        if (n / 5 > 0):
            return "V" * (n/5) + int_to_minimal_roman(n%5)
        return "I" * n

    def tokenize(inStr):
        intersperse = inStr.replace("", " ")[1:-1]
        return intersperse.split(" ")

    def minimumForm(inStr):
        return int_to_minimal_roman(roman_to_int(tokenize(inStr)))
        
    def main():
        f = open('roman.txt', 'r')
        originalCount, finalCount, totalCount = 0, 0, 0
        for line in f.read().split('\n'):
            originalCount += len(line)
            finalCount += len(minimumForm(line))

        print originalCount - finalCount
        
    if __name__ == "__main__":
        main()

def problem95():
    """
    Project Euler, problem 95. Amicable chains.
    A pair of numbers (a,b) is amicable if the sum of the divisors of a is equal
    to b. An amicable chain is one a chain that pairs back to its starting point

    Find the smallest member of the longest amicable chain with no element
    exceeding one million.
    """
    import time
    begin = time.time()
    LIMIT = 1000000

    memo = {1:[None]} #hash of numbers mapped to their divisors

    for i in xrange(2, LIMIT+1): #all numbers are divisible by 1
        memo[i] = [1]

    for i in xrange(2, (LIMIT+1)/2): #update memo with divisors of memo[i]
        step = i
        count = i + i
        while count < LIMIT+1:
            memo[count].append(i)
            count += i
    print "Divisors hash created."

    memoSum = {1:None} #this is a hash of sum of divisors
    
    for i in xrange(2, LIMIT+1):
        memoSum[i] = sum(x for x in memo[i]) #map number to sum of its divisors

    print "Sum of divisors hash created"
    def insertN(n, listA):
        return [n] + listA

    def createChain(n):
        chain = []
        temp = memoSum[n]
        while temp not in chain:
            if temp == 1:
                return None
            elif temp > LIMIT:
                return None
            elif temp == n:
                return insertN(n, chain)
            chain.append(temp)
            temp = memoSum[temp]
        return None

    memoMin = {} #hash mapping chain length to min value of chain

    for i in xrange(100): #assuming max chain < 100
        memoMin[i] = []

    longestChain = []
    for i in xrange(2, LIMIT+1):
        temp = createChain(i)
        if temp != None:
            if len(temp) >= len(longestChain):
                print "longest chain so far: ", temp, len(temp)
                longestChain = temp
                smallest = min(i, min(temp))
                memoMin[len(temp)].append(smallest)
                answerLength = len(temp)

    print "The length of the longest amicable chain was ", answerLength
    print "smallest member of this chain was ", min(memoMin[answerLength])

    end = time.time()
    print (end-begin), "seconds"

def problem99():
    """
    Comparing two numbers written in index form like 2^11 and 3^7 is not
    difficult, as any calculator would confirm 2^11 = 2048 < 3^7 = 2187.

    However, confirming that 632382^518061 > 519432^525806 would be much more
    difficult, as both numbers contain over three million digits.

    Using base_exp.txt, a 22K text file containing one thousand lines with a
    base/exponent pair on each line, determine which line number has the
    greatest numerical value.
    """

    # Gonna use the power rule of logarithms, where log(x^y) = y*log(x)
    # to reduce the size of the numbers.

    from urllib import urlopen
    from numpy import log #base e
    from time import clock

    f = urlopen('http://projecteuler.net/project/base_exp.txt', 'r').readlines()
    maxval, bestbase, bestexp, besti = 0, 0, 0, 0
    BASE, EXP = 0, 1
    begin = clock()
    for i, line in enumerate(f):
        data = tuple(int(i) for i in line.split(','))
        current = data[EXP]*log(data[BASE]) #power rule
        if current > maxval:
            maxval = current
            bestbase = data[BASE]
            bestexp = data[EXP]
            besti = i
            
    print "The best line is: ", besti+1
    print bestbase, "^", bestexp
    print clock() - begin
