from sys.info import simdwidthof, is_apple_silicon, num_logical_cores
from memory import memcpy, memcmp, memset_zero, UnsafePointer
from algorithm import vectorize, parallelize
from buffer import Buffer, NDBuffer, DimList
from algorithm.reduction import sum, cumsum, variance
from collections import InlinedFixedVector, Dict, Optional
import math
import random
from random import random_float64
from sklmini_mo.utility.utils import cov_value, gauss_jordan, add, sub, mul, div
from sklmini_mo.utility import matmul
from python import Python, PythonObject

struct Matrix(Stringable, Writable):
    var height: Int
    var width: Int
    var size: Int
    var data: UnsafePointer[Float64]
    var order: String
    alias simd_width: Int = 4 * simdwidthof[DType.float64]() if is_apple_silicon() else 2 * simdwidthof[DType.float64]()

    # initialize from UnsafePointer
    @always_inline
    fn __init__(out self, height: Int, width: Int, data: UnsafePointer[Float64] = UnsafePointer[Float64](), order: String = 'c'):
        self.height = height
        self.width = width
        self.size = height * width
        self.data = UnsafePointer[Float64].alloc(self.size)
        self.order = order.lower()
        if data:
            memcpy(self.data, data, self.size)

    # initialize from List
    fn __init__(out self, height: Int, width: Int, def_input: List[Float64]):
        self.height = height
        self.width = width
        self.size = height * width
        self.data = UnsafePointer[Float64].alloc(self.size)
        self.order = 'c'
        if len(def_input) > 0:
            memcpy(self.data, def_input.data, self.size)

    # initialize from list object
    fn __init__(out self, height: Int, width: Int, def_input: object) raises:
        self.height = height
        self.width = width
        self.size = height * width
        self.data = UnsafePointer[Float64].alloc(self.size)
        self.order = 'c'
        var rng: Int = len(def_input)
        for i in range(rng):
            self.data[i] = atof(str(def_input[i])).cast[DType.float64]()

    # initialize in 2D numpy style
    fn __init__(out self, npstyle: String, order: String = 'c') raises:
        var mat = npstyle.replace(' ', '')
        if mat[0] == '[' and mat[1] == '[' and mat[len(mat) - 1] == ']' and mat[len(mat) - 2] == ']':
            self.width = 0
            self.size = 0
            self.data = UnsafePointer[Float64]()
            self.order = order.lower()
            var rows = mat[:-1].split(']')
            self.height = len(rows) - 1
            for i in range(self.height):
                var values = rows[i][2:].split(',')
                if i == 0:
                    self.width = len(values)
                    self.size = self.height * self.width
                    self.data = UnsafePointer[Float64].alloc(self.size)
                for j in range(self.width):
                    self.store[1](i, j, atof(values[j]).cast[DType.float64]())
        else:
            raise Error('Error: Matrix is not initialized in the correct form!')

    @staticmethod
    fn rand(height: Int, width: Int) -> Matrix:
        var mat = Matrix(height, width)
        @parameter
        fn random_init_mat(index: Int):
            mat.data[index] = random_float64()
        parallelize[random_init_mat](mat.size, num_logical_cores())
        return mat^

    fn __copyinit__(out self, other: Self):
        self.height = other.height
        self.width = other.width
        self.size = other.size
        self.data = UnsafePointer[Float64].alloc(self.size)
        self.order = other.order
        memcpy(self.data, other.data, self.size)

    fn __moveinit__(out self, owned existing: Self):
        self.height = existing.height
        self.width = existing.width
        self.size = existing.size
        self.data = existing.data
        self.order = existing.order
        existing.height = existing.width = existing.size = 0
        existing.order = ''
        existing.data = UnsafePointer[Float64]()

    @always_inline
    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float64, nelts]:
        var loc: Int
        if self.order == 'c':
            loc = (y * self.width) + x
        else:
            loc = (x * self.height) + y
        return self.data.load[width=nelts](loc)

    @always_inline
    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float64, nelts]):
        var loc: Int
        if self.order == 'c':
            loc = (y * self.width) + x
        else:
            loc = (x * self.height) + y
        return self.data.store(loc, val)

    # access an element
    @always_inline
    fn __getitem__(self, row: Int, column: Int) raises -> Float64:
        var loc: Int
        if self.order == 'c':
            loc = (row * self.width) + column
        else:
            loc = (column * self.height) + row
        if loc > self.size - 1:
            raise Error("Error: Location is out of range!")
        return self.data[loc]

    # access a row
    @always_inline
    fn __getitem__(self, row: Int) raises -> Matrix:
        if row >= self.height or row < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' or self.height == 1:
            return Matrix(1, self.width, self.data + (row * self.width), self.order)
        var mat = Matrix(1, self.width, order= self.order)
        var tmpPtr = self.data + row
        @parameter
        fn convert[simd_width: Int](idx: Int):
            mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.height))
            tmpPtr += simd_width * self.height
        vectorize[convert, self.simd_width](mat.width)
        return mat^

    # access a row (unsafe)
    @always_inline
    fn __getitem__(self, row: Int, *, unsafe: Bool) -> Matrix:
        if self.order == 'c' or self.height == 1:
            return Matrix(1, self.width, self.data + (row * self.width), self.order)
        var mat = Matrix(1, self.width, order= self.order)
        var tmpPtr = self.data + row
        @parameter
        fn convert[simd_width: Int](idx: Int):
            mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.height))
            tmpPtr += simd_width * self.height
        vectorize[convert, self.simd_width](mat.width)
        return mat^

    # access a row with offset
    @always_inline
    fn __getitem__(self, row: Int, offset: Bool, start_i: Int) raises -> Matrix:
        if row >= self.height or row < 0 or start_i >= self.width or start_i < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' or self.height == 1:
            return Matrix(1, self.width - start_i, self.data + (row * self.width) + start_i, self.order)
        var mat = Matrix(1, self.width - start_i, order= self.order)
        var tmpPtr = self.data + row + (start_i * self.height)
        @parameter
        fn convert[simd_width: Int](idx: Int):
            mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.height))
            tmpPtr += simd_width * self.height
        vectorize[convert, self.simd_width](mat.width)
        return mat^

    # access a column
    @always_inline
    fn __getitem__(self, row: String, column: Int) raises -> Matrix:
        if column >= self.width or column < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' and self.width > 1:
            var mat = Matrix(self.height, 1)
            var tmpPtr = self.data + column
            @parameter
            fn convert[simd_width: Int](idx: Int):
                mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.width))
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](mat.height)
            return mat^
        return Matrix(self.height, 1, self.data + (column * self.height), self.order)

    # access a column (unsafe)
    @always_inline
    fn __getitem__(self, row: String, column: Int, *, unsafe: Bool) -> Matrix:
        if self.order == 'c' and self.width > 1:
            var mat = Matrix(self.height, 1)
            var tmpPtr = self.data + column
            @parameter
            fn convert[simd_width: Int](idx: Int):
                mat.data.store(idx, tmpPtr.strided_load[width=simd_width](self.width))
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](mat.height)
            return mat^
        return Matrix(self.height, 1, self.data + (column * self.height), self.order)

    # access given rows (by their indices)
    @always_inline
    fn __getitem__(self, rows: Matrix) raises -> Matrix:
        var mat = Matrix(rows.size, self.width, order= self.order)
        for i in range(rows.size):
            mat[i] = self[int(rows.data[i])]
        return mat^

    # access given columns (by their indices)
    @always_inline
    fn __getitem__(self, row: String, columns: Matrix) raises -> Matrix:
        var mat = Matrix(self.height, columns.size, order= self.order)
        for i in range(columns.size):
            mat[row, i] = self[row, int(columns.data[i])]
        return mat^

    # access given rows (by their indices)
    @always_inline
    fn __getitem__(self, rows: List[Int]) raises -> Matrix:
        var mat = Matrix(len(rows), self.width, order= self.order)
        for i in range(mat.height):
            mat[i] = self[rows[i]]
        return mat^

    # access given columns (by their indices)
    @always_inline
    fn __getitem__(self, row: String, columns: List[Int]) raises -> Matrix:
        var mat = Matrix(self.height, len(columns), order= self.order)
        for i in range(mat.width):
            mat[row, i] = self[row, columns[i]]
        return mat^

    # Slicing (numpy style)
    # TODO: Fix the slicing Currently, it is not working as expected
    @always_inline
    fn __getitem__(self, owned row_slice: Slice, col: Int) -> Self:
        return self.__getitem__(row_slice, slice(col,col+1))

    @always_inline
    fn __getitem__(self, row: Int, owned col_slice: Slice) -> Self:
        return self.__getitem__(slice(row,row+1),col_slice)

    @always_inline
    fn __getitem__(self, owned row_slice: Slice, owned col_slice: Slice) -> Self:
        self._adjust_row_slice_(row_slice)
        self._adjust_col_slice_(col_slice)

        var row_length = self.get_slice_length(row_slice)
        var col_length = self.get_slice_length(col_slice, 'col')
        
        var dest_mat = Self(row_length, col_length)
        
        alias simd_width: Int = self.simd_width
        
        # Calculate the actual indices for the slices
        var row_start = row_slice.start.value()
        var col_start = col_slice.start.value()
        var step = col_slice.step if col_slice.step else 1

        for idx_rows in range(row_length):
            var src_ptr = self.data.offset((row_start + idx_rows) * self.width + col_start)
            @parameter
            fn slice_col_vectorize[simd_width: Int](idx: Int) -> None:
                if idx + simd_width <= col_length:
                    dest_mat.data.store(
                        idx + idx_rows * col_length,
                        src_ptr.load(step.value())
                    )
                else:
                    # Handle remaining elements
                    for j in range(idx, col_length):
                        dest_mat.data[idx + idx_rows * col_length] = src_ptr.load[width=1](j * step.value())
            
            vectorize[slice_col_vectorize, simd_width](col_length)
        
        return dest_mat

    @always_inline
    fn _adjust_row_slice_(self, inout span: Slice):
        if not span.start:
            span.start = 0
        if span.start.value() < 0:
            span.start = self.height + span.start.value()    
        if not span.end:
            span.end = self.height
        elif span.end.value() < 0:
            span.end = self.height + span.end.value()
        if span.end.value() > self.height:
            span.end = self.height

    @always_inline
    fn _adjust_col_slice_(self, inout span: Slice):
        if not span.start:
            span.start = 0
        if span.start.value() < 0:
            span.start = self.width + span.start.value()
        if not span.end:
            span.end = self.width
        elif span.end.value() < 0:
            span.end = self.width + span.end.value()
        if span.end.value() > self.width:
            span.end = self.width

    @always_inline
    fn get_slice_length(self, span: Slice, type: String = 'row') -> Int:
        var start = span.start.value() if span.start else 0
        var end: Int
        if type == 'row':
            end = span.end.value() if span.end else self.height
        else:  # col
            end = span.end.value() if span.end else self.width
        return end - start

    # altenative slicing (rows only)
    @always_inline
    fn get_slice(self, start: Int, end: Int) raises -> Matrix:
        if start >= self.height or end > self.height:
            raise Error("Slice indices out of range")
            
        var mat = Matrix(end - start, self.width)
        if self.order == 'c':
            for i in range(end - start):
                for j in range(self.width):
                    mat.data[i * self.width + j] = self.data[(i + start) * self.width + j]
        else:
            for i in range(end - start):
                for j in range(self.width):
                    mat.data[i * self.width + j] = self.data[(i + start) + j * self.height]
        return mat^

    # replace an element
    @always_inline
    fn __setitem__(inout self, row: Int, column: Int, val: Float64) raises:
        var loc: Int
        if self.order == 'c':
            loc = (row * self.width) + column
        else:
            loc = (column * self.height) + row
        if loc > self.size - 1:
            raise Error("Error: Location is out of range!")
        self.data[loc] = val
    
    # replace the given row
    @always_inline
    fn __setitem__(inout self, row: Int, val: Matrix) raises:
        if row >= self.height or row < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' or self.height == 1:
            memcpy(self.data + (row * self.width), val.data, val.size)
        else:
            var tmpPtr = self.data + row
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.height)
                tmpPtr += simd_width * self.height
            vectorize[convert, self.simd_width](val.size)

    # replace the given row (unsafe)
    @always_inline
    fn __setitem__(inout self, row: Int, val: Matrix, *, unsafe: Bool):
        if self.order == 'c' or self.height == 1:
            memcpy(self.data + (row * self.width), val.data, val.size)
        else:
            var tmpPtr = self.data + row
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.height)
                tmpPtr += simd_width * self.height
            vectorize[convert, self.simd_width](val.size)

    # replace the given row with offset
    @always_inline
    fn __setitem__(inout self, row: Int, offset: Bool, start_i: Int, val: Matrix) raises:
        if row >= self.height or row < 0 or start_i >= self.width or start_i < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' or self.height == 1:
            memcpy(self.data + (row * self.width) + start_i, val.data, val.size)
        else:
            var tmpPtr = self.data + row + (start_i * self.height)
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.height)
                tmpPtr += simd_width * self.height
            vectorize[convert, self.simd_width](val.size)

    # replace the given column
    @always_inline
    fn __setitem__(inout self, row: String, column: Int, val: Matrix) raises:
        if column >= self.width or column < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' and self.width > 1:
            var tmpPtr = self.data + column
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.width)
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](val.size)
        else:
            memcpy(self.data + (column * self.height), val.data, val.size)

    # replace the given column (unsafe)
    @always_inline
    fn __setitem__(inout self, row: String, column: Int, val: Matrix, *, unsafe: Bool):
        if self.order == 'c' and self.width > 1:
            var tmpPtr = self.data + column
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.width)
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](val.size)
        else:
            memcpy(self.data + (column * self.height), val.data, val.size)

    # replace the given column with offset
    @always_inline
    fn __setitem__(inout self, offset: Bool, start_i: Int, column: Int, val: Matrix) raises:
        if column >= self.width or column < 0 or start_i >= self.height or start_i < 0:
            raise Error("Error: Index out of range!")
        if self.order == 'c' and self.width > 1:
            var tmpPtr = self.data + column + (start_i * self.width)
            @parameter
            fn convert[simd_width: Int](idx: Int):
                tmpPtr.strided_store[width=simd_width](val.data.load[width=simd_width](idx), self.width)
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](val.size)
        else:
            memcpy(self.data + (column * self.height) + start_i, val.data, val.size)

    # replace given rows (by their indices)
    @always_inline
    fn __setitem__(inout self, rows: Matrix, rhs: Matrix) raises:
        for i in range(rows.size):
            self[int(rows.data[i])] = rhs[i]

    # replace given columns (by their indices)
    @always_inline
    fn __setitem__(inout self, row: String, columns: Matrix, rhs: Matrix) raises:
        for i in range(columns.size):
            self[row, int(columns.data[i])] = rhs[row, i]

    @always_inline
    fn __del__(owned self):
        if self.data:
            self.data.free()

    @always_inline
    fn __len__(self) -> Int:
        return self.size

    @always_inline
    fn __eq__(self, rhs: Float64) -> InlinedFixedVector[Bool]:
        var result = InlinedFixedVector[Bool](self.size)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] == rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] == rhs
            parallelize[cmp](self.size, num_logical_cores())
        return result^

    @always_inline
    fn __ne__(self, rhs: Float64) -> InlinedFixedVector[Bool]:
        var result = InlinedFixedVector[Bool](self.size)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] != rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] != rhs
            parallelize[cmp](self.size, num_logical_cores())
        return result^

    @always_inline
    fn __gt__(self, rhs: Float64) -> InlinedFixedVector[Bool]:
        var result = InlinedFixedVector[Bool](self.size)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] > rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] > rhs
            parallelize[cmp](self.size, num_logical_cores())
        return result^

    @always_inline
    fn __ge__(self, rhs: Float64) -> InlinedFixedVector[Bool]:
        var result = InlinedFixedVector[Bool](self.size)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] >= rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] >= rhs
            parallelize[cmp](self.size, num_logical_cores())
        return result^

    @always_inline
    fn __lt__(self, rhs: Float64) -> InlinedFixedVector[Bool]:
        var result = InlinedFixedVector[Bool](self.size)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] < rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] < rhs
            parallelize[cmp](self.size, num_logical_cores())
        return result^

    @always_inline
    fn __le__(self, rhs: Float64) -> InlinedFixedVector[Bool]:
        var result = InlinedFixedVector[Bool](self.size)
        if self.size < 131072:
            for i in range(self.size):
                result[i] = self.data[i] <= rhs
        else:
            @parameter
            fn cmp(i: Int):
                result[i] = self.data[i] <= rhs
            parallelize[cmp](self.size, num_logical_cores())
        return result^

    @always_inline
    fn __eq__(self, rhs: Self) -> Bool:
        return self.height == rhs.height and self.width == rhs.width and memcmp(self.data, rhs.data, self.size) == 0

    @always_inline
    fn __ne__(self, rhs: Self) -> Bool:
        return not self == rhs

    @always_inline
    fn __add__(self, rhs: Self) raises -> Self:
        if self.height == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self + rhs.data[0]
            if self.width == 1:
                return self.data[0] + rhs
            if self.width == rhs.width:
                return self._broadcast_row(rhs.height, self.width, rhs.order)._elemwise_matrix[add](rhs)
            raise Error("Error: Cannot add matrices with different shapes!")
        if self.width == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self + rhs.data[0]
            if self.height == rhs.height:
                return self._broadcast_column(self.height, rhs.width, rhs.order)._elemwise_matrix[add](rhs)
            raise Error("Error: Cannot add matrices with different shapes!")
        if rhs.height == 1:
            if rhs.width == 1:
                return self + rhs.data[0]
            elif rhs.width == self.width:
                return self._elemwise_matrix[add](rhs._broadcast_row(self.height, self.width, self.order))
            raise Error("Error: Cannot add matrices with different shapes!")
        if rhs.width == 1:
            if rhs.height == self.height:
                return self._elemwise_matrix[add](rhs._broadcast_column(self.height, self.width, self.order))
            raise Error("Error: Cannot add matrices with different shapes!")
        if self.height == rhs.height and self.width == rhs.width:
            if self.order == rhs.order:
                return self._elemwise_matrix[add](rhs)
            return self._elemwise_matrix[add](rhs.asorder(self.order))
        raise Error("Error: Cannot add matrices with different shapes!")

    @always_inline
    fn __iadd__(inout self, rhs: Self) raises:
        self = self + rhs

    @always_inline
    fn __add__(self, rhs: Float64) -> Self:
        return self._elemwise_scalar[add](rhs)

    @always_inline
    fn __radd__(self, lhs: Float64) -> Self:
        return self + lhs

    @always_inline
    fn __iadd__(inout self, rhs: Float64):
        self = self + rhs

    @always_inline
    fn __sub__(self, rhs: Self) raises -> Self:
        if self.height == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self - rhs.data[0]
            if self.width == 1:
                return self.data[0] - rhs
            if self.width == rhs.width:
                return self._broadcast_row(rhs.height, self.width, rhs.order)._elemwise_matrix[sub](rhs)
            raise Error("Error: Cannot subtract matrices with different shapes!")
        if self.width == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self - rhs.data[0]
            if self.height == rhs.height:
                return self._broadcast_column(self.height, rhs.width, rhs.order)._elemwise_matrix[sub](rhs)
            raise Error("Error: Cannot subtract matrices with different shapes!")
        if rhs.height == 1:
            if rhs.width == 1:
                return self - rhs.data[0]
            elif rhs.width == self.width:
                return self._elemwise_matrix[sub](rhs._broadcast_row(self.height, self.width, self.order))
            raise Error("Error: Cannot subtract matrices with different shapes!")
        if rhs.width == 1:
            if rhs.height == self.height:
                return self._elemwise_matrix[sub](rhs._broadcast_column(self.height, self.width, self.order))
            raise Error("Error: Cannot subtract matrices with different shapes!")
        if self.height == rhs.height and self.width == rhs.width:
            if self.order == rhs.order:
                return self._elemwise_matrix[sub](rhs)
            return self._elemwise_matrix[sub](rhs.asorder(self.order))
        raise Error("Error: Cannot subtract matrices with different shapes!")

    @always_inline
    fn __isub__(inout self, rhs: Self) raises:
        self = self - rhs

    @always_inline
    fn __sub__(self, rhs: Float64) -> Self:
        return self._elemwise_scalar[sub](rhs)

    @always_inline
    fn __rsub__(self, lhs: Float64) -> Self:
        return -(self - lhs)

    @always_inline
    fn __isub__(inout self, rhs: Float64):
        self = self - rhs

    @always_inline
    fn __truediv__(self, rhs: Self) raises -> Self:
        if self.height == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self / rhs.data[0]
            if self.width == 1:
                return self.data[0] / rhs
            if self.width == rhs.width:
                return self._broadcast_row(rhs.height, self.width, rhs.order)._elemwise_matrix[div](rhs)
            raise Error("Error: Cannot divide matrices with different shapes!")
        if self.width == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self / rhs.data[0]
            if self.height == rhs.height:
                return self._broadcast_column(self.height, rhs.width, rhs.order)._elemwise_matrix[div](rhs)
            raise Error("Error: Cannot divide matrices with different shapes!")
        if rhs.height == 1:
            if rhs.width == 1:
                return self / rhs.data[0]
            elif rhs.width == self.width:
                return self._elemwise_matrix[div](rhs._broadcast_row(self.height, self.width, self.order))
            raise Error("Error: Cannot divide matrices with different shapes!")
        if rhs.width == 1:
            if rhs.height == self.height:
                return self._elemwise_matrix[div](rhs._broadcast_column(self.height, self.width, self.order))
            raise Error("Error: Cannot divide matrices with different shapes!")
        if self.height == rhs.height and self.width == rhs.width:
            if self.order == rhs.order:
                return self._elemwise_matrix[div](rhs)
            return self._elemwise_matrix[div](rhs.asorder(self.order))
        raise Error("Error: Cannot divide matrices with different shapes!")

    @always_inline
    fn __itruediv__(inout self, rhs: Self) raises:
        self = self / rhs

    @always_inline
    fn __truediv__(self, rhs: Float64) -> Self:
        return self._elemwise_scalar[div](rhs)

    @always_inline
    fn __rtruediv__(self, lhs: Float64) -> Self:
        return lhs * (self ** -1)

    @always_inline
    fn __itruediv__(inout self, rhs: Float64):
        self = self / rhs

    @always_inline
    fn __mul__(self, rhs: Self) raises -> Self:
        if self.width != rhs.height:
            raise Error('Error: Cannot multiply matrices with shapes (' + str(self.height) + ', ' + str(self.width) + ') and (' + str(rhs.height) + ', ' + str(rhs.width) + ')')
        var A = matmul.Matrix[DType.float64](self.data, (self.height, self.width))
        var B = matmul.Matrix[DType.float64](rhs.data, (rhs.height, rhs.width))
        var C = matmul.Matrix[DType.float64]((self.height, rhs.width))
        memset_zero(C.data, self.height * rhs.width)
        matmul.matmul(self.height, self.width, rhs.width, C, A, B)
        var mat = Matrix(self.height, rhs.width)
        mat.data = C.data
        return mat^

    @always_inline
    fn __imul__(inout self, rhs: Self) raises:
        self = self * rhs

    @always_inline
    fn __mul__(self, rhs: Float64) -> Self:
        return self._elemwise_scalar[mul](rhs)

    @always_inline
    fn __rmul__(self, lhs: Float64) -> Self:
        return self * lhs

    @always_inline
    fn __imul__(inout self, rhs: Float64):
        self = self * rhs

    @always_inline
    fn __neg__(self) -> Self:
        return self * (-1.0)

    @always_inline
    fn __pow__(self, p: Int) -> Self:
        if p == 1:
            return self
        var mat = Self(self.height, self.width, order= self.order)
        if self.size < 262144:
            @parameter
            fn math_vectorize[simd_width: Int](idx: Int):
                mat.data.store(idx, pow(self.data.load[width=simd_width](idx), p))
            vectorize[math_vectorize, self.simd_width](self.size)
        else:
            var n_vects = int(math.ceil(self.size / self.simd_width))
            @parameter
            fn math_vectorize_parallelize(i: Int):
                var idx = i * self.simd_width
                mat.data.store(idx, pow(self.data.load[width=self.simd_width](idx), p))
            parallelize[math_vectorize_parallelize](n_vects, num_logical_cores())
        return mat^

    @always_inline
    fn __ipow__(inout self, rhs: Int):
        self = self ** rhs

    @always_inline
    fn ele_mul(self, rhs: Matrix) raises -> Matrix:
        # element-wise multiplication
        if self.height == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self * rhs.data[0]
            if self.width == 1:
                return self.data[0] * rhs
            if self.width == rhs.width:
                return self._broadcast_row(rhs.height, self.width, rhs.order)._elemwise_matrix[mul](rhs)
            raise Error("Error: Cannot element-wise multiply matrices with different shapes!")
        if self.width == 1:
            if rhs.height == 1 and rhs.width == 1:
                return self * rhs.data[0]
            if self.height == rhs.height:
                return self._broadcast_column(self.height, rhs.width, rhs.order)._elemwise_matrix[mul](rhs)
            raise Error("Error: Cannot element-wise multiply matrices with different shapes!")
        if rhs.height == 1:
            if rhs.width == 1:
                return self * rhs.data[0]
            elif rhs.width == self.width:
                return self._elemwise_matrix[mul](rhs._broadcast_row(self.height, self.width, self.order))
            raise Error("Error: Cannot element-wise multiply matrices with different shapes!")
        if rhs.width == 1:
            if rhs.height == self.height:
                return self._elemwise_matrix[mul](rhs._broadcast_column(self.height, self.width, self.order))
            raise Error("Error: Cannot element-wise multiply matrices with different shapes!")
        if self.height == rhs.height and self.width == rhs.width:
            if self.order == rhs.order:
                return self._elemwise_matrix[mul](rhs)
            return self._elemwise_matrix[mul](rhs.asorder(self.order))
        raise Error("Error: Cannot element-wise multiply matrices with different shapes!")

    @always_inline
    fn where(self, cmp: InlinedFixedVector[Bool], _true: Float64, _false: Float64) -> Matrix:
        var mat = Matrix(self.height, self.width, order= self.order)
        if self.size < 40960:
            for i in range(self.size):
                if cmp[i]:
                    mat.data[i] = _true
                else:
                    mat.data[i] = _false
        else:
            @parameter
            fn p(i: Int):
                if cmp[i]:
                    mat.data[i] = _true
                else:
                    mat.data[i] = _false
            parallelize[p](self.size, num_logical_cores())
        return mat^

    fn where(self, cmp: InlinedFixedVector[Bool], _true: Matrix, _false: Float64) -> Matrix:
        var mat = Matrix(self.height, self.width, order= self.order)
        if self.size < 40960:
            for i in range(self.size):
                if cmp[i]:
                    mat.data[i] = _true.data[i]
                else:
                    mat.data[i] = _false
        else:
            @parameter
            fn p(i: Int):
                if cmp[i]:
                    mat.data[i] = _true.data[i]
                else:
                    mat.data[i] = _false
            parallelize[p](self.size, num_logical_cores())
        return mat^

    fn where(self, cmp: InlinedFixedVector[Bool], _true: Float64, _false: Matrix) -> Matrix:
        var mat = Matrix(self.height, self.width, order= self.order)
        if self.size < 40960:
            for i in range(self.size):
                if cmp[i]:
                    mat.data[i] = _true
                else:
                    mat.data[i] = _false.data[i]
        else:
            @parameter
            fn p(i: Int):
                if cmp[i]:
                    mat.data[i] = _true
                else:
                    mat.data[i] = _false.data[i]
            parallelize[p](self.size, num_logical_cores())
        return mat^

    @always_inline
    fn argwhere(self, cmp: InlinedFixedVector[Bool]) -> Matrix:
        var args = List[Float64]()
        for i in range(self.size):
            if cmp[i]:
                args.append(i // self.width)
                args.append(i % self.width)
        return Matrix(len(args) // 2, 2, args)

    @always_inline
    fn argwhere_l(self, cmp: InlinedFixedVector[Bool]) -> List[Int]:
        var args = List[Int]()
        for i in range(self.size):
            if cmp[i]:
                args.append(i)
        return args^

    @always_inline
    fn C_transpose(self) -> Matrix:
        var mat = Matrix(self.width, self.height)
        for idx_col in range(self.width):
            var tmpPtr = self.data + idx_col
            @parameter
            fn convert[simd_width: Int](idx: Int):
                mat.data.store(idx + idx_col * self.height, tmpPtr.strided_load[width=simd_width](self.width))
                tmpPtr += simd_width * self.width
            vectorize[convert, self.simd_width](self.height)
        return mat^

    @always_inline
    fn F_transpose(self) -> Matrix:
        var mat = Matrix(self.width, self.height, order= self.order)
        for idx_row in range(self.height):
            var tmpPtr = self.data + idx_row
            @parameter
            fn convert[simd_width: Int](idx: Int):
                mat.data.store(idx + idx_row * self.width, tmpPtr.strided_load[width=simd_width](self.height))
                tmpPtr += simd_width * self.height
            vectorize[convert, self.simd_width](self.width)
        return mat^
    
    @always_inline
    fn T(self) raises -> Matrix:
        if self.height == 1 or self.width == 1:
            return self.reshape(self.width, self.height)
        if self.order == 'c':
            return self.C_transpose()
        return self.F_transpose()

    fn asorder(self, order: String) raises -> Matrix:
        _order = order.lower()
        if _order == self.order:
            return self
        var mat = self.T().reshape(self.height, self.width)
        mat.order = _order
        return mat^

    @always_inline
    fn cumsum(self) -> Matrix:
        var mat = Matrix(self.height, self.width, order= self.order)
        cumsum(Buffer[DType.float64](mat.data, self.size), Buffer[DType.float64](self.data, self.size))
        return mat^

    @always_inline
    fn sum(self) -> Float64:
        try: 
            return sum(Buffer[DType.float64](self.data, self.size))
        except:
            return 0.0

    @always_inline
    fn sum(self, axis: Int) -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True].sum()
            else:
                @parameter
                fn p0(i: Int):
                    mat.data[i] = self['', i, unsafe=True].sum()
                parallelize[p0](self.width, num_logical_cores())
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True].sum()
            else:
                @parameter
                fn p1(i: Int):
                    mat.data[i] = self[i, unsafe=True].sum()
                parallelize[p1](self.height, num_logical_cores())
        return mat^

    @always_inline
    fn mean(self) -> Float64:
        return self.sum() / self.size

    @always_inline
    fn mean(self, axis: Int) -> Matrix:
        if axis == 0:
            return self.sum(0) / self.height
        return self.sum(1) / self.width

    fn mean_slow(self) -> Float64:
        return (self / self.size).sum()

    fn mean_slow0(self) -> Matrix:
        var mat = Matrix(1, self.width, order= self.order)
        if self.width < 768:
            for i in range(self.width):
                mat.data[i] = self['', i, unsafe=True].mean_slow()
        else:
            @parameter
            fn p0(i: Int):
                mat.data[i] = self['', i, unsafe=True].mean_slow()
            parallelize[p0](self.width, num_logical_cores())
        return mat^

    @always_inline
    fn _var(self) -> Float64:
        return ((self - self.mean()) ** 2).mean()

    @always_inline
    fn _var(self, _mean: Float64) -> Float64:
        return ((self - _mean) ** 2).mean()

    @always_inline
    fn _var(self, axis: Int) -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True]._var()
            else:
                @parameter
                fn p0(i: Int):
                    mat.data[i] = self['', i, unsafe=True]._var()
                parallelize[p0](self.width, num_logical_cores())
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True]._var()
            else:
                @parameter
                fn p1(i: Int):
                    mat.data[i] = self[i, unsafe=True]._var()
                parallelize[p1](self.height, num_logical_cores())
        return mat^

    @always_inline
    fn _var(self, axis: Int, _mean: Matrix) -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True]._var(_mean.data[i])
            else:
                @parameter
                fn p0(i: Int):
                    mat.data[i] = self['', i, unsafe=True]._var(_mean.data[i])
                parallelize[p0](self.width, num_logical_cores())
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True]._var(_mean.data[i])
            else:
                @parameter
                fn p1(i: Int):
                    mat.data[i] = self[i, unsafe=True]._var(_mean.data[i])
                parallelize[p1](self.height, num_logical_cores())
        return mat^

    @always_inline
    fn std(self) -> Float64:
        return math.sqrt(self._var())

    @always_inline
    fn std(self, _mean: Float64) -> Float64:
        return math.sqrt(self._var(_mean))

    fn std_slow(self, _mean: Float64) -> Float64:
        return math.sqrt(((self - _mean) ** 2).mean_slow())

    @always_inline
    fn std(self, axis: Int) -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True].std()
            else:
                @parameter
                fn p0(i: Int):
                    mat.data[i] = self['', i, unsafe=True].std()
                parallelize[p0](self.width, num_logical_cores())
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True].std()
            else:
                @parameter
                fn p1(i: Int):
                    mat.data[i] = self[i, unsafe=True].std()
                parallelize[p1](self.height, num_logical_cores())
        return mat^

    @always_inline
    fn std(self, axis: Int, _mean: Matrix) -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True].std(_mean.data[i])
            else:
                @parameter
                fn p0(i: Int):
                    mat.data[i] = self['', i, unsafe=True].std(_mean.data[i])
                parallelize[p0](self.width, num_logical_cores())
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True].std(_mean.data[i])
            else:
                @parameter
                fn p1(i: Int):
                    mat.data[i] = self[i, unsafe=True].std(_mean.data[i])
                parallelize[p1](self.height, num_logical_cores())
        return mat^

    fn std_slow(self, axis: Int, _mean: Matrix) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True].std_slow(_mean.data[i])
            else:
                @parameter
                fn p0(i: Int):
                    mat.data[i] = self['', i, unsafe=True].std_slow(_mean.data[i])
                parallelize[p0](self.width, num_logical_cores())
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True].std_slow(_mean.data[i])
            else:
                @parameter
                fn p1(i: Int):
                    mat.data[i] = self[i, unsafe=True].std_slow(_mean.data[i])
                parallelize[p1](self.height, num_logical_cores())
        return mat^

    @always_inline
    fn abs(self) -> Matrix:
        var mat = Matrix(self.height, self.width, order= self.order)
        if self.size < 262144:
            @parameter
            fn math_vectorize[simd_width: Int](idx: Int):
                mat.data.store(idx, abs(self.data.load[width=simd_width](idx)))
            vectorize[math_vectorize, self.simd_width](self.size)
        else:
            var n_vects = int(math.ceil(self.size / self.simd_width))
            @parameter
            fn math_vectorize_parallelize(i: Int):
                var idx = i * self.simd_width
                mat.data.store(idx, abs(self.data.load[width=self.simd_width](idx)))
            parallelize[math_vectorize_parallelize](n_vects, num_logical_cores())
        return mat^

    @always_inline
    fn log(self) -> Matrix:
        return self._elemwise_math[math.log]()

    @always_inline
    fn sqrt(self) -> Matrix:
        return self._elemwise_math[math.sqrt]()

    @always_inline
    fn exp(self) -> Matrix:
        return self._elemwise_math[math.exp]()

    @always_inline
    fn min(self) -> Float64:
        try :
            return algorithm.reduction.min(Buffer[DType.float64](self.data, self.size))
        except:
            return 0.0

    @always_inline
    fn min(self, axis: Int) -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True].min()
            else:
                @parameter
                fn p0(i: Int):
                    mat.data[i] = self['', i, unsafe=True].min()
                parallelize[p0](self.width, num_logical_cores())
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True].min()
            else:
                @parameter
                fn p1(i: Int):
                    mat.data[i] = self[i, unsafe=True].min()
                parallelize[p1](self.height, num_logical_cores())
        return mat^

    @always_inline
    fn max(self) -> Float64:
        try:
            return algorithm.reduction.max(Buffer[DType.float64](self.data, self.size))
        except:
            return 0.0

    @always_inline
    fn max(self, axis: Int) -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(1, self.width, order= self.order)
            if self.width < 768:
                for i in range(self.width):
                    mat.data[i] = self['', i, unsafe=True].max()
            else:
                @parameter
                fn p0(i: Int):
                    mat.data[i] = self['', i, unsafe=True].max()
                parallelize[p0](self.width, num_logical_cores())
        elif axis == 1:
            mat = Matrix(self.height, 1, order= self.order)
            if self.height < 768:
                for i in range(self.height):
                    mat.data[i] = self[i, unsafe=True].max()
            else:
                @parameter
                fn p1(i: Int):
                    mat.data[i] = self[i, unsafe=True].max()
                parallelize[p1](self.height, num_logical_cores())
        return mat^

    fn argmin(self, axis: Optional[Int] = None) raises -> Matrix:
        if axis == None:
            var min_val = self.data[0]
            var min_idx = 0
            for i in range(self.size):
                if self.data[i] < min_val:
                    min_val = self.data[i]
                    min_idx = i
            var result = Matrix(1, 2)
            result[0, 0] = Float64(min_idx // self.width)
            result[0, 1] = Float64(min_idx % self.width)
            return result
            
        elif axis == 0:  # Along columns
            var result = Matrix(1, self.width)
            for j in range(self.width):
                var min_val = self[0, j]
                var min_idx = 0
                for i in range(self.height):
                    if self[i, j] < min_val:
                        min_val = self[i, j]
                        min_idx = i
                result[0, j] = Float64(min_idx)
            return result
            
        elif axis == 1:  # Along rows
            var result = Matrix(self.height, 1)
            for i in range(self.height):
                var min_val = self[i, 0]
                var min_idx = 0
                for j in range(self.width):
                    if self[i, j] < min_val:
                        min_val = self[i, j]
                        min_idx = j
                result[i, 0] = Float64(min_idx)
            return result
            
        else:
            raise Error("Axis must be None, 0, or 1")

    fn argmax(self, axis: Optional[Int] = None) raises -> Matrix:
        if axis == None:
            var max_val = self.data[0]
            var max_idx = 0
            @parameter
            fn p(i: Int):
                if self.data[i] > max_val:
                    max_val = self.data[i]
                    max_idx = i
            parallelize[p](self.size, num_logical_cores())
            var result = Matrix(1, 2)
            result[0, 0] = Float64(max_idx // self.width)
            result[0, 1] = Float64(max_idx % self.width)
            return result
            
        elif axis == 0:  # Along columns
            var result = Matrix(1, self.width)
            for j in range(self.width):
                var max_val = self[0, j]
                var max_idx = 0
                for i in range(self.height):
                    if self[i, j] > max_val:
                        max_val = self[i, j]
                        max_idx = i
                result[0, j] = Float64(max_idx)
            return result
            
        elif axis == 1:  # Along rows
            var result = Matrix(self.height, 1)
            for i in range(self.height):
                var max_val = self[i, 0]
                var max_idx = 0
                for j in range(self.width):
                    if self[i, j] > max_val:
                        max_val = self[i, j]
                        max_idx = j
                result[i, 0] = Float64(max_idx)
            return result
            
        else:
            raise Error("Axis must be None, 0, or 1")

    @always_inline
    fn reshape(self, height: Int, width: Int) raises -> Matrix:
        """Reshape with one dimension automatically calculated."""
        if height == -1:
            if self.size % width != 0:
                raise Error("Cannot reshape: incompatible dimensions")
            return self._reshape(self.size // width, width)
        elif width == -1:
            if self.size % height != 0:
                raise Error("Cannot reshape: incompatible dimensions")
            return self._reshape(height, self.size // height)
        else:
            return self._reshape(height, width)

    fn _reshape(self, height: Int, width: Int) raises -> Matrix:
        """Reshape matrix to new dimensions while preserving data."""
        # Check if reshape is valid
        if height * width != self.size:
            raise Error("Cannot reshape array of size " + str(self.size) + 
                    " into shape (" + str(height) + ", " + str(width) + ")")
        
        # Create new matrix with desired shape
        var mat = Matrix(height, width, order=self.order)
        
        # Copy data preserving order
        if self.order == 'c':
            # Row-major order
            for i in range(self.size):
                var old_row = i // self.width
                var old_col = i % self.width
                var new_row = i // width
                var new_col = i % width
                mat[new_row, new_col] = self[old_row, old_col]
        else:
            # Column-major order
            for i in range(self.size):
                var old_col = i // self.height
                var old_row = i % self.height
                var new_col = i // height
                var new_row = i % height
                mat[new_row, new_col] = self[old_row, old_col]
                
        return mat^
    
    fn cov(self) raises -> Matrix:
        var c = Matrix(self.height, self.height, order= self.order)
        for i in range(self.height):
            for j in range(self.height):
                c[i, j] = cov_value(self[j], self[i])
        return c^

    fn inv(self) raises -> Matrix:
        if self.height != self.width:
            raise Error("Error: Matrix must be square to inverse!")
        var tmp = gauss_jordan(self.concatenate(Matrix.eye(self.height, self.order), 1))
        var mat = Matrix(self.height, self.height, order= self.order)
        for i in range(tmp.height):
            mat[i] = tmp[i, True, tmp[i].size//2]
        return mat^

    @staticmethod
    @always_inline
    fn eye(n: Int, order: String = 'c') -> Matrix:
        var result = Matrix.zeros(n, n, order)
        var tmpPtr = result.data
        @parameter
        fn convert[simd_width: Int](idx: Int):
            tmpPtr.strided_store[width=simd_width](1.0, (n + 1))
            tmpPtr += simd_width * (n + 1)
        vectorize[convert, result.simd_width](n)
        return result^

    @always_inline
    fn norm(self) -> Float64:
        return math.sqrt((self ** 2).sum())

    @always_inline
    fn qr(self, standard: Bool = False) raises -> Tuple[Matrix, Matrix]:
        # QR decomposition. standard: make R diag positive
        # Householder algorithm, i.e., not Gram-Schmidt or Givens
        # if not square, verify m greater-than n ("tall")
        # if standard==True verify m == n

        var Q = Matrix.eye(self.height, self.order)
        var R = self
        var end: Int
        if self.height == self.width:
            end = self.width - 1
        else:
            end = self.width
        for i in range(end):
            var H = Matrix.eye(self.height, self.order)
            # -------------------
            var a: Matrix = R[True, i, i]  # partial column vector
            var norm_a: Float64 = a.norm()
            if a.data[0] < 0.0: norm_a = -norm_a
            var v: Matrix = a / (a.data[0] + norm_a)
            v.data[0] = 1.0
            var h = Matrix.eye(a.height, self.order)  # H reflection
            h -= (2 / (v.T() * v))[0, 0] * (v * v.T())
            # -------------------
            for j in range(H.height - i):
                H[j + i, True, i] = h[j]  # copy h into H
            Q = Q * H
            R = H * R

        if standard:  # A must be square
            var S = Matrix.zeros(self.width, self.width, order= self.order)  # signs of diagonal
            for i in range(self.width):
                if R[i, i] < 0.0:
                    S[i, i] = -1.0
                else:
                    S[i, i] = 1.0
            Q = Q * S
            R = S * R

        return Q^, R^

    @always_inline
    fn is_upper_tri(self, tol: Float64) raises -> Bool:
        for i in range(self.height):
            for j in range(i):
                if abs(self[i, j]) > tol:
                    return False
        return True

    fn eigen(self, max_ct: Int = 10000) raises -> Tuple[Matrix, Matrix]:
        var X = self
        var pq = Matrix.eye(self.height, self.order)

        var ct: Int = 0
        while ct < max_ct:
            var Q: Matrix
            var R: Matrix
            Q, R = X.qr()
            pq = pq * Q  # accum Q
            X = R * Q
            ct += 1

            if X.is_upper_tri(1.0e-8):
                break

        if ct == max_ct:
            print("WARN (eigen): no converge!")

        # eigenvalues are diag elements of X
        var e_vals = Matrix.zeros(1, self.height, order= self.order)
        for i in range(self.height):
            e_vals.data[i] = X[i, i]

        # eigenvectors are columns of pq
        var e_vecs = pq

        return e_vals^, e_vecs^

    fn outer(self, rhs: Matrix) raises -> Matrix:
        var mat = Matrix(self.size, rhs.size, order= self.order)
        if mat.order == 'c':
            for i in range(mat.height):
                mat[i] = self.data[i] * rhs
        else:
            for i in range(mat.width):
                mat['', i] = self * rhs.data[i]
        return mat^

    fn concatenate(self, rhs: Matrix, axis: Int) raises -> Matrix:
        var mat = Matrix(0, 0)
        if axis == 0:
            mat = Matrix(self.height + rhs.height, self.width, order= self.order)
            if self.order == 'c' or self.height == 1:
                memcpy(mat.data, self.data, self.size)
                memcpy(mat.data + self.size, rhs.data, rhs.size)
            else:
                for i in range(self.width):
                    memcpy(mat.data + i * mat.height, self.data + i * self.height, self.height)
                    memcpy(mat.data + i * mat.height + self.height, rhs.data + i * rhs.height, rhs.height)
        elif axis == 1:
            mat = Matrix(self.height, self.width + rhs.width, order= self.order)
            if self.order == 'c' and self.width > 1:
                for i in range(self.height):
                    memcpy(mat.data + i * mat.width, self.data + i * self.width, self.width)
                    memcpy(mat.data + i * mat.width + self.width, rhs.data + i * rhs.width, rhs.width)
            else:
                memcpy(mat.data, self.data, self.size)
                memcpy(mat.data + self.size, rhs.data, rhs.size)
        return mat^

    @always_inline
    fn bincount(self) -> InlinedFixedVector[Int]:
        var freq = Dict[Int, Int]()
        var max_val: Int = 0
        for i in range(self.size):
            var d = int(self.data[i])
            try:
                freq[d] += 1
            except:
                freq[d] = 1
                if d > max_val:
                    max_val = d

        var vect = InlinedFixedVector[Int](capacity = max_val + 1)
        for i in range(max_val + 1):
            try:
                vect[i] = freq[i]
            except:
                vect[i] = 0
        return vect^

    @staticmethod
    @always_inline
    fn unique(data: PythonObject) raises -> Tuple[List[String], List[Int]]:
        var list = List[String]()
        var freq = List[Int]()
        var rng = len(data)
        for i in range(rng):
            var d = str(data[i])
            if d in list:
                freq[list.index(d)] += 1
            else:
                list.append(d)
                freq.append(1)
        return list^, freq^

    @always_inline
    fn unique(self) -> Dict[Int, Int]:
        var freq = Dict[Int, Int]()
        for i in range(self.size):
            var d = int(self.data[i])
            try:
                freq[d] += 1
            except:
                freq[d] = 1
        return freq^

    @always_inline
    fn uniquef(self) -> List[Float64]:
        var list = List[Float64]()
        for i in range(self.size):
            var contains = False
            for j in list:
                if j[] == self.data[i]:
                    contains = True
            if not contains:
                list.append(self.data[i])
        return list^

    @staticmethod
    @always_inline
    fn zeros(height: Int, width: Int, order: String = 'c') -> Matrix:
        var mat = Matrix(height, width, order= order)
        memset_zero(mat.data, mat.size)
        return mat^

    @staticmethod
    @always_inline
    fn ones(height: Int, width: Int, order: String = 'c') -> Matrix:
        return Matrix.full(height, width, 1.0, order)

    @staticmethod
    fn full(height: Int, width: Int, val: Float64, order: String = 'c') -> Matrix:
        var mat = Matrix(height, width, order= order)
        mat.fill(val)
        return mat^

    @always_inline
    fn fill_zero(self):
        memset_zero(self.data, self.size)

    @always_inline
    fn fill(self, val: Float64):
        Buffer[DType.float64](self.data, self.size).fill(val)

    @staticmethod
    @always_inline
    fn random(height: Int, width: Int, order: String = 'c') -> Matrix:
        random.seed()
        var mat = Matrix(height, width, order= order)
        random.rand(mat.data, mat.size, min=0.0, max=1.0)
        return mat^

    @staticmethod
    @always_inline
    fn rand_choice(arang: Int, size: Int, replace: Bool = True) -> List[Int]:
        random.seed()
        var result = UnsafePointer[Int].alloc(size)
        if not replace:
            for i in range(size):
                result[i] = i
        for i in range(size - 1, 0, -1):
            if not replace:
                # Fisher-Yates shuffle
                var j = int(random.random_ui64(0, i))
                result[i], result[j] = result[j], result[i]
            else:
                result[i] = int(random.random_ui64(0, arang - 1))
        return List[Int](ptr=result, length=size, capacity=size)

    @staticmethod
    @always_inline
    fn rand_choice(arang: Int, size: Int, replace: Bool, seed: Int) -> List[Int]:
        random.seed(seed)
        var result = UnsafePointer[Int].alloc(size)
        if not replace:
            for i in range(size):
                result[i] = i
        for i in range(size - 1, 0, -1):
            if not replace:
                # Fisher-Yates shuffle
                var j = int(random.random_ui64(0, i))
                result[i], result[j] = result[j], result[i]
            else:
                result[i] = int(random.random_ui64(0, arang - 1))
        return List[Int](ptr=result, length=size, capacity=size)

    @staticmethod
    fn from_numpy(np_arr: PythonObject, order: String = 'c') raises -> Matrix:
        var np = Python.import_module("numpy")
        # Ensure contiguous array with correct type and order
        var np_arr_f = np.ascontiguousarray(np_arr, dtype='float64')
        
        # Get dimensions
        var height = int(np_arr_f.shape[0])
        var width = 1
        if len(np_arr_f.shape) > 1:
            width = int(np_arr_f.shape[1])
        
        # Create new Matrix with proper dimensions
        var mat = Matrix(height, width, order=order)
        
        # Get pointer to numpy array data
        var data_ptr = np_arr_f.__array_interface__['data'][0].unsafe_get_as_pointer[DType.float64]()
        
        # Copy data
        memcpy(mat.data, data_ptr, height * width)
        
        return mat^

    fn to_numpy(self) raises -> PythonObject:
        var np = Python.import_module("numpy")
        var np_arr = np.empty((self.height,self.width), dtype='f', order= self.order.upper())
        memcpy(np_arr.__array_interface__['data'][0].unsafe_get_as_pointer[DType.float64](), self.data, self.size)
        return np_arr^

    @always_inline
    fn _broadcast_row(self, height: Int, width: Int, order: String) -> Matrix:
        var mat = Matrix(height, width, order=order)
        if height * width < 262144 and height < 1024:
            for i in range(mat.height):
                mat[i, unsafe=True] = self
        else:
            @parameter
            fn broadcast(i: Int):
                mat[i, unsafe=True] = self
            parallelize[broadcast](mat.height, num_logical_cores())
        return mat^

    @always_inline
    fn _broadcast_column(self, height: Int, width: Int, order: String) -> Matrix:
        var mat = Matrix(height, width, order=order)
        if height * width < 262144 and width < 1024:
            for i in range(mat.width):
                mat['', i, unsafe=True] = self
        else:
            @parameter
            fn broadcast(i: Int):
                mat['', i, unsafe=True] = self
            parallelize[broadcast](mat.width, num_logical_cores())
        return mat^

    @always_inline
    fn _elemwise_scalar[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, rhs: Float64) -> Self:
        var mat = Matrix(self.height, self.width, order= self.order)
        if self.size < 262144:
            @parameter
            fn scalar_vectorize[simd_width: Int](idx: Int):
                mat.data.store(idx, func[DType.float64, simd_width](self.data.load[width=simd_width](idx), rhs))
            vectorize[scalar_vectorize, self.simd_width](self.size)
        else:
            var n_vects = int(math.ceil(self.size / self.simd_width))
            @parameter
            fn scalar_vectorize_parallelize(i: Int):
                var idx = i * self.simd_width
                mat.data.store(idx, func[DType.float64, self.simd_width](self.data.load[width=self.simd_width](idx), rhs))
            parallelize[scalar_vectorize_parallelize](n_vects, num_logical_cores())
        return mat^

    @always_inline
    fn _elemwise_matrix[func: fn[dtype: DType, width: Int](SIMD[dtype, width],SIMD[dtype, width])->SIMD[dtype, width]](self, rhs: Self) -> Self:
        var mat = Matrix(self.height, self.width, order= self.order)
        if self.size < 262144:
            @parameter
            fn matrix_vectorize[simd_width: Int](idx: Int):
                mat.data.store(idx, func[DType.float64, simd_width](self.data.load[width=simd_width](idx), rhs.data.load[width=simd_width](idx)))
            vectorize[matrix_vectorize, self.simd_width](self.size)
        else:
            var n_vects = int(math.ceil(self.size / self.simd_width))
            @parameter
            fn matrix_vectorize_parallelize(i: Int):
                var idx = i * self.simd_width
                mat.data.store(idx, func[DType.float64, self.simd_width](self.data.load[width=self.simd_width](idx), rhs.data.load[width=self.simd_width](idx)))
            parallelize[matrix_vectorize_parallelize](n_vects, num_logical_cores())
        return mat^

    @always_inline
    fn _elemwise_math[func: fn[dtype: DType, width: Int](SIMD[dtype, width])->SIMD[dtype, width]](self) -> Self:
        var mat = Matrix(self.height, self.width, order= self.order)
        if self.size < 262144:
            @parameter
            fn math_vectorize[simd_width: Int](idx: Int):
                mat.data.store(idx, func(self.data.load[width=simd_width](idx)))
            vectorize[math_vectorize, self.simd_width](self.size)
        else:
            var n_vects = int(math.ceil(self.size / self.simd_width))
            @parameter
            fn math_vectorize_parallelize(i: Int):
                var idx = i * self.simd_width
                mat.data.store(idx, func(self.data.load[width=self.simd_width](idx)))
            parallelize[math_vectorize_parallelize](n_vects, num_logical_cores())
        return mat^

    fn write_to[W: Writer](self, mut writer: W):
        var res: String = "["
        var strings = List[String]()
        for i in range(self.width):
            var max_len: Int = 0
            for j in range(self.height):
                strings.append("")
                var val = self.load[1](j, i)
                if val >= 0:
                    strings[j] += " "
                strings[j] += str(val)
                if len(strings[j]) > max_len:
                    max_len = len(strings[j])
            for j in range(self.height):
                var rng: Int = max_len - len(strings[j]) + 1
                for _ in range(rng):
                    strings[j] += " "

        for i in range(self.height):
            if i != 0:
                res += " "
            res += "[" + strings[i] + "]"
            if i != self.height - 1:
                res += "\n"
        writer.write(res + "]")

    fn __str__(self) -> String:
        return String.write(self)