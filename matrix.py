class matrix(object):
    array = []
    row = 0
    col = 0
    
    def __init__(self, array, isFloat = False):
        #if [0][0] is float, rest is probably float (no map)
        self.isFloat = isFloat
        if (isFloat):
            self.array = array
        else:
            self.array = [[float(element) for element in row] for row in array]
            isFloat = True
        self.row = len(array)
        self.col = len(array[0])
    
    def __repr__(self):
        string = 'matrix\n['
        for i in range(self.row):
            string += (str(self.array[i]) + '\n')
        return string[:-1] +']'
    
    #helper func: epsilon rounding to prettify output
    def epsilon(self):
        mArr = self.array
        acc = 5 #accuracy to x significant figures
        ep = 0.0000001 #abitrary num, smaller than x sig figs
        return matrix([[round(num - ep, acc) for num in row] for row in mArr], True)

    #add matrices
    def __add__(self, other):
        assert (self.row == other.row and self.col == other.col), "Can only add matrices with same dimensions"
        return matrix([[i+j for i,j in zip(x,y)] for (x,y) in zip(self.array, other.array)], True)

    #subtract matrices
    def __sub__(self, other):
        assert (self.row == other.row and self.col == other.col), "Can only subtract matrices with same dimensions"
        return matrix([[i-j for i,j in zip(x,y)] for (x,y) in zip(self.array, other.array)], True)
    
    #multiply matrices or scalar with matrix      
    def __mul__(self, other):
        #handle scalar multiplication first
        if (type(other) in [int, float]):
            return matrix([[other * num for num in row] for row in self.array], True)
        assert (self.row == other.col), "Mismatch row and columns"
        #matrix * matrix
        product = []
        for rowcounter in range(self.row):
            singlerow = []
            for colcounter in range(other.col):
                cell = 0
                for i in range(other.row):
                    cell += (self.array[rowcounter][i] * other.array[i][colcounter])
                singlerow.append(cell)
            product.append(singlerow)
        return matrix(product, True)

    #help define for scalar on left side of multiplication
    #order of (matrix A * matrix B) wont be affected since it'll use top def
    __rmul__=__mul__
    
    #switch rows and columns (transpose a matrix)        
    def transpose(self):
        return matrix([list(i) for i in zip(*self.array)])
    
    #create identity matrix for square array
    def identity(self):
        assert (self.col == self.row), "Not a square matrix"
        I = [[1.0 if x == y else 0.0 for x in range(self.col)] for y in range(self.col)]
        return matrix(I, True)
    
    #find inverse of matrix via jordan-gauss elimination
    #apply complimentary operations to ID matrix to transforms
    #into inverse matrix (if inverse exists)
    def inverse(self):
        ID = matrix.identity(self).array
        mArr = self.array[:]
        
        #turn to echelon form:
        #checks to see if rows below first col is zeroed out
        #ifnot: sub scalar multiple of topmost echelon form row to zero out
        # c_ (subscript are the complimentary operations to ID matrix)
        def echelon(arr, comp):
            for row in range(1,self.row):
                for j in range(row, self.row):
                    if (arr[j][row - 1]):
                        scalar = (arr[j][row - 1]/arr[row-1][row-1])
                        temp = [scalar * a for a in arr[row - 1]]
                        arr[j] = [(a - b) for a, b in zip(arr[j], temp)]
                        c_temp = [scalar * a for a in comp[row - 1]]
                        comp[j] = [(a - b) for a, b in zip(comp[j], c_temp)]
            return arr, comp

        #scale diagonals to 1's
        def scale(arr, comp):
            for row in range(self.row):
                lead = arr[row][row]
                if (lead != 1):
                    scalar = 1/lead
                    arr[row] = [(scalar * a) for a in arr[row]]
                    comp[row] = [(scalar * a) for a in comp[row]]
            return arr, comp

        #turn rest into ID matrix
        def b_echelon(arr, comp):
            for row in range(self.row - 2, -1, -1):
                for j in range(self.row - 1, row, -1):
                    if (arr[row][j]):
                        scalar = (arr[row][j])
                        temp = [scalar * a for a in arr[j]]
                        arr[row] = [(a - b) for a, b in zip(arr[row], temp)]
                        c_temp = [scalar * a for a in comp[j]]
                        comp[row] = [(a - b) for a, b in zip(comp[row], c_temp)]
            return comp
        
        
        mArr, ID = echelon(mArr, ID)
        mArr, ID = scale(mArr, ID)
        ID = b_echelon(mArr, ID)

        #return after scaled through epsilon rounding
        return matrix(ID).epsilon()

#-----------------------------------------------------------------------------------------------------
#Matrix class ends here, and Kalman Filter Begins
#-----------------------------------------------------------------------------------------------------