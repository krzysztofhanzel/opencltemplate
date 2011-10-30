using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using OpenCLTemplate;

namespace OpenCLTemplate.LinearAlgebra
{
    /// <summary>Encapsulates functions to create a symmetric, positive definite matrix</summary>
    public class floatSymPosDefMatrix
    {
        #region Constructors
        /// <summary>Constructor.</summary>
        /// <param name="n">Number of matrix rows (n x n)</param>
        public floatSymPosDefMatrix(int n)
        {
            this.N = n;
            values = new float[(n * (n + 1)) >> 1];

            LocalInitCL();
        }

        /// <summary>Constructor.</summary>
        /// <param name="vals">Matrix elements. Length should be n*(n+1)/2 where matrix is nxn</param>
        public floatSymPosDefMatrix(float[] vals)
        {
            int temp = (int)Math.Floor(Math.Sqrt(1 + (vals.Length << 3)));
            int n = temp * temp == 1 + (vals.Length << 3) ? (temp - 1) >> 1 : temp >> 1;

            if (vals.Length != (n * (n + 1)) >> 1) throw new Exception("Invalid vector length");

            values = (float[])vals.Clone();
            this.N = n;

            LocalInitCL();
        }
        #endregion

        #region Matrix information

        /// <summary>Matrix dimension</summary>
        private int N;

        /// <summary>Matrix values</summary>
        private float[] values;

        /// <summary>Access matrix elements</summary>
        /// <param name="i">Row index of element to access</param>
        /// <param name="j">Column index of element to access</param>
        public float this[int i, int j]
        {
            get
            {
                if (i >= N || j >= N) throw new Exception("Index out of bounds");

                if (i >= j) return values[((i * (i + 1)) >> 1) + j];
                else return values[((j * (j + 1)) >> 1) + i];
            }
            set
            {
                if (i >= N || j >= N) throw new Exception("Index out of bounds");

                IsCholeskyFactorized = false;
                IsMatrixInClMemoryUpdated = false;
                if (i >= j) values[((i * (i + 1)) >> 1) + j] = value;
                else values[((j * (j + 1)) >> 1) + i] = value;
            }
        }

        /// <summary>Returns a string representing this instance</summary>
        public override string ToString()
        {
            int maxN = 200;
            string s = "";

            for (int i = 0; i < Math.Min(maxN,this.N); i++)
            {
                for (int j = 0; j < Math.Min(maxN, this.N); j++)
                {
                    s += (this[i, j]).ToString() + "\t\t";
                }
                s += "\n";
            }

            return s;
        }

        #endregion

        #region Cholesky factorization
        /// <summary>Was the matrix cholesky factorized since last update?</summary>
        private bool IsCholeskyFactorized = false;
        /// <summary>Cholesky factorization</summary>
        private float[] cholDec;

        /// <summary>Computes Cholesky factorization of a matrix</summary>
        public void ComputeCholesky()
        {
            if (CLCalc.CLAcceleration != CLCalc.CLAccelerationType.UsingCL || this.N < 120)
            {
                NoCLCholesky();
                if (CLCalc.CLAcceleration == CLCalc.CLAccelerationType.UsingCL) CLcholDec.WriteToDevice(cholDec);
            }
            else CLCholesky();
        }
        
        /// <summary>Naive computation of the Cholesky factorization for very small systems or systems without OpenCL</summary>
        private void NoCLCholesky()
        {
            cholDec = new float[(N * (N + 1)) >> 1];
            float[] thisCp = (float[])this.values.Clone();

            float[] prevVals = new float[N];

            float temp;
            float temp2;
            int indTemp;

            for (int i = 0; i < N; i++)
            {
                //pivot
                temp = 1.0f / (float)Math.Sqrt(thisCp[((i * (i + 1)) >> 1) + i]);

                //Row elements
                for (int j = i; j < N; j++)
                {
                    indTemp = ((j * (j + 1)) >> 1) + i;
                    temp2 = temp * thisCp[indTemp];

                    cholDec[indTemp] = temp2;
                    prevVals[j] = temp2;
                }

                //Global update
                for (int p = i + 1; p < N; p++)
                {
                    int pp = ((p * (p + 1)) >> 1);
                    for (int q = i + 1; q <= p; q++)
                    {
                        indTemp = pp + q;
                        thisCp[indTemp] = thisCp[indTemp] - prevVals[p] * prevVals[q];
                    }
                }
            }

            IsCholeskyFactorized = true;
        }

        /// <summary>Returns true if Cholesky decomposition succeeded to 10*float.Epsilon precision</summary>
        private bool CheckDecomposition()
        {
            if (!IsCholeskyFactorized) throw new Exception("Matrix not factorized");

            double check = 0;
            for (int i = 0; i < N; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double elem = 0;
                    for (int k = 0; k <= j; k++)
                        elem += cholDec[((i * (i + 1)) >> 1) + k] * cholDec[((j * (j + 1)) >> 1) + k];

                    double dif = (this.values[((i * (i + 1)) >> 1) + j] - elem) / elem;
                    check += Math.Abs(dif);
                }
            }
            check /= (double)(0.5 * N * (N + 1));
            return (check < 5E-5);
        }

        #region OpenCL Cholesky decomposition, OpenCL sources

        /// <summary>Cholesky decomposition in Device memory</summary>
        CLCalc.Program.Variable CLcholDec;
        /// <summary>Copy of values of this matrix</summary>
        CLCalc.Program.Variable CLthisCp;
        /// <summary>Is matrix in OpenCL memory updated?</summary>
        private bool IsMatrixInClMemoryUpdated = false;

        /// <summary>Cholesky elements computed in previous step</summary>
        CLCalc.Program.Variable CLprevVals;
        /// <summary>Offsets to perform calculations</summary>
        CLCalc.Program.Variable CLoffSet;

        /// <summary>B variable during back/forward subst</summary>
        CLCalc.Program.Variable CLb;
        /// <summary>Y variable during back/forward subst</summary>
        CLCalc.Program.Variable CLy;


        /// <summary>Cholesky update kernel</summary>
        private static CLCalc.Program.Kernel kernelCholeskyUpdt;
        /// <summary>Rescale remanining values</summary>
        private static CLCalc.Program.Kernel kernelCholeskyRescale;
        /// <summary>Rescale remanining values without __constant</summary>
        private static CLCalc.Program.Kernel kernelCholeskyReScal2;

        /// <summary>Computes one term in backsubstitution</summary>
        private static CLCalc.Program.Kernel kernelComputeTerm;
        /// <summary>Updates vector during forward substitution</summary>
        private static CLCalc.Program.Kernel kernelUpdateBForward;
        /// <summary>Updates vector during backward substitution</summary>
        private static CLCalc.Program.Kernel kernelUpdateYBackward;

        /// <summary>Matrix - vector product</summary>
        private static CLCalc.Program.Kernel kernelMatrVecMultiply;
        /// <summary>Computes AtWA</summary>
        private static CLCalc.Program.Kernel kernelComputeAtWA;

        #region OpenCL Source
        private class CholeskySrc
        {
            public string srcCholesky = @"

__kernel void CholeskyUpdt(__global const float * thisCp,
                           __global       float * cholDec,
                           __global       float * prevVals,
                           __constant     int   * offset)
{
   int i = offset[0];
   int j = get_global_id(0)+i;
   
   int indTemp = ((j * (j + 1)) >> 1) + i;
   
   float temp2 = (thisCp[indTemp]) * rsqrt(thisCp[((i * (i + 1)) >> 1) + i]);
   
   cholDec[indTemp] = temp2;
   prevVals[j] = temp2;
}

__kernel void CholeskyReScale(__global       float * thisCp,
                              __global const float * cholDec,
                              __constant     float * prevVals,
                              __constant     int   * offset)
{
   int idx = get_global_id(0);
   int temp = (int)floor(sqrt(1.0f + (idx << 3)));
   
   int p = (temp - 1) >> 1;
   int q = idx - ((p * (p + 1)) >> 1);
   
   p += offset[0]; q += offset[0];
   
   int indTemp = ((p * (p + 1)) >> 1) + q;

   thisCp[indTemp] = thisCp[indTemp] - prevVals[p] * prevVals[q];
}

__kernel void CholeskyReScal2(__global       float * thisCp,
                              __global const float * cholDec,
                              __global const float * prevVals,
                              __constant     int   * offset)
{
   int idx = get_global_id(0);
   int temp = (int)floor(sqrt(1.0f + (idx << 3)));
   
   int p = (temp - 1) >> 1;
   int q = idx - ((p * (p + 1)) >> 1);
   
   p += offset[0]; q += offset[0];
   
   int indTemp = ((p * (p + 1)) >> 1) + q;
   thisCp[indTemp] = thisCp[indTemp] - prevVals[p] * prevVals[q];
}

";

            public string srcBkSubs = @"
__kernel void ComputeTerm(__global const float * cholDec,
                          __global       float * resp,
                          __global const float * b,
                          __constant     int   * offset)
{
   //worksize = 1
   int i = offset[0];
   resp[i] = b[i] / cholDec[((i * (i + 1)) >> 1) + i];
}

__kernel void UpdateBForward(__global const float * cholDec,
                             __global const float * y,
                             __global       float * b,
                             __constant     int   * offset)
{
   int i = offset[0];
   int j = get_global_id(0) + i + 1;
   b[j] -= cholDec[((j * (j + 1)) >> 1) + i] * y[i];
}

__kernel void UpdateYBackward(__global const float * cholDec,
                              __global const float * resp,
                              __global       float * y,
                              __constant     int   * offset)
{
   int i = offset[0];
   int j = get_global_id(0);
   y[j] -= cholDec[((i * (i + 1)) >> 1) + j] * resp[i];
}
";

            public string srcOperations = @"
__kernel void MatrVecMultiply(__global const float * SymM,
                              __global const float * v,
                              __global       float * x)
{
   //global_size = n
   int n = get_global_size(0);
   int p = get_global_id(0);
   
   float val = 0;
   for (int k = 0; k < p; k++)
   {
       val += SymM[((p*(1+p))>>1)+k]*v[k];
   }
   for (int k = p; k < n; k++)
   {
       val += SymM[((k*(1+k))>>1)+p]*v[k];
   }
   
   x[p] = val;
}


__kernel void ComputeAtWA(__global const float * A,
                          __global const float * W,
                          __global       float * AtWA,
                          __constant     int *   dimsA)
                          
{
   //global_work_size = n(n+1)/2
   int idx = get_global_id(0);
   int temp = (int)floor(sqrt(1.0f + (idx << 3)));
   
   int p = (temp - 1) >> 1;
   int q = idx - ((p * (p + 1)) >> 1);
   
   int m = dimsA[0];
   int n = dimsA[1];
   
   float val = 0;
   int nk;
   for (int k = 0; k < m; k++)
   {
      nk = n*k;
      val += A[p + nk]*A[q + nk]*W[k];
   }
   
   AtWA[((p*(1+p))>>1)+q] = val;
}
";
        #endregion


        }

        static floatSymPosDefMatrix()
        {
            if (CLCalc.CLAcceleration == CLCalc.CLAccelerationType.Unknown) CLCalc.InitCL();

            if (CLCalc.CLAcceleration == CLCalc.CLAccelerationType.UsingCL)
            {
                if (kernelCholeskyUpdt == null)
                {
                    CholeskySrc src = new CholeskySrc();
                    CLCalc.Program.Compile(new string[] { src.srcCholesky, src.srcBkSubs, src.srcOperations });

                    kernelCholeskyUpdt = new CLCalc.Program.Kernel("CholeskyUpdt");
                    kernelCholeskyRescale = new CLCalc.Program.Kernel("CholeskyReScale");
                    kernelCholeskyReScal2 = new CLCalc.Program.Kernel("CholeskyReScal2");

                    kernelComputeTerm = new CLCalc.Program.Kernel("ComputeTerm");
                    kernelUpdateBForward = new CLCalc.Program.Kernel("UpdateBForward");
                    kernelUpdateYBackward = new CLCalc.Program.Kernel("UpdateYBackward");

                    kernelMatrVecMultiply = new CLCalc.Program.Kernel("MatrVecMultiply");
                    kernelComputeAtWA = new CLCalc.Program.Kernel("ComputeAtWA");
                }
            }
        }

        private void LocalInitCL()
        {
            if (CLCalc.CLAcceleration == CLCalc.CLAccelerationType.Unknown) CLCalc.InitCL();

            if (CLCalc.CLAcceleration == CLCalc.CLAccelerationType.UsingCL)
            {
                CLoffSet = new CLCalc.Program.Variable(new int[1]);
                CLthisCp = new CLCalc.Program.Variable(this.values);
                CLcholDec = new CLCalc.Program.Variable(this.values);
                CLprevVals = new CLCalc.Program.Variable(new float[N]);

                CLb = new CLCalc.Program.Variable(new float[N]);
                CLy = new CLCalc.Program.Variable(new float[N]);
            }
        }

        /// <summary>Cholesky decomposition using OpenCL</summary>
        private void CLCholesky()
        {
            CLCalc.Program.Kernel kernelRescale = (64 * 1024 / sizeof(float) >= N) ? kernelCholeskyRescale : kernelCholeskyReScal2;

            cholDec = new float[(N * (N + 1)) >> 1];

            //Copies values to device memory
            float[] thisCp = (float[])this.values.Clone();

            if (!IsMatrixInClMemoryUpdated)
            {
                CLthisCp.WriteToDevice(this.values);
                IsMatrixInClMemoryUpdated = true;
            }

            float[] prevVals = new float[N];
            
            int[] offset = new int[1];
            CLoffSet.WriteToDevice(offset);

            //kernel arguments
            CLCalc.Program.Variable[] args = new CLCalc.Program.Variable[] { CLthisCp, CLcholDec, CLprevVals, CLoffSet };

            for (int i = 0; i < N; i++)
            {
                //Update cholesky decomposition matrix
                offset[0] = i;
                kernelCholeskyUpdt.Execute(args, N - i);

                ////DEBUG
                //CLcholDec.ReadFromDeviceTo(cholDec);

                //n = get_global_size(0)+offset[0]; globalsize = N-i

                //Update component values
                offset[0] = i + 1;
                CLoffSet.WriteToDevice(offset);

                int nnItems = ((N - i) * (N - i - 1)) >> 1; //get_global_size(0)
                if (nnItems > 0) kernelCholeskyRescale.Execute(args, nnItems);

                ////DEBUG
                //CLthisCp.ReadFromDeviceTo(thisCp);

                #region Index access test
                //for (int p = i + 1; p < N; p++)
                //{
                //    for (int q = i + 1; q <= p; q++)
                //    {
                //        int nItems = ((N - i) * (N - i - 1)) >> 1; //get_global_size(0)

                //        int idx = (((p - i - 1) * (p - i)) >> 1) + (q - i - 1);
                //        int temp = (int)Math.Floor(Math.Sqrt(1 + (idx << 3)));
                //        int pp = (temp - 1) >> 1;//temp * temp == 1 + (idx << 3) ? (temp - 1) >> 1 : temp >> 1;
                //        int qq = idx - ((pp * (pp + 1)) >> 1);

                //        pp += i + 1; //+offset[0]
                //        qq += i + 1; //+offset[0]

                //        if (pp != p || qq != q)
                //        {
                //        }
                //    }
                //}
                #endregion
            }

            CLcholDec.ReadFromDeviceTo(cholDec);
            IsCholeskyFactorized = true;

        }
        #endregion
        #endregion

        #region Linear system solving, determinant

        /// <summary>Solves system Ax = b and returns x</summary>
        /// <param name="b">b vector</param>
        /// <param name="refine">Refine solution? Recommended: true</param>
        public float[] LinearSolve(float[] b, bool refine)
        {
            //System.Diagnostics.Stopwatch swChol = new System.Diagnostics.Stopwatch();
            //System.Diagnostics.Stopwatch swResto = new System.Diagnostics.Stopwatch();
            //System.Diagnostics.Stopwatch swLinSolve = new System.Diagnostics.Stopwatch();

            if (b.Length != N) throw new Exception("Dimensions not compatible");
            
            //swChol.Start();
            if (!IsCholeskyFactorized) ComputeCholesky();
            //swChol.Stop();

            //swResto.Start();

            //Computes preliminar solution
            float[] resp = linsolve(b); //linsolveCL too slow
            if (!refine)
            {
                //swResto.Stop();
                return resp;
            }

            float[] deltab = new float[N];
            double totalRes;

            for (int iter = 0; iter < 10; iter++)
            {
                //Computes residues
                totalRes = 0;
                for (int i = 0; i < N; i++)
                {
                    double residue = 0;
                    for (int k = 0; k < N; k++)
                    {
                        residue += this[i, k] * resp[k];
                    }
                    residue -= b[i];
                    deltab[i] = (float)residue;
                    totalRes += Math.Abs(residue);
                }
                totalRes /= (double)N;

                if (totalRes < 1E-6) iter = 10;
                else
                {
                    //swLinSolve.Start();
                    float[] deltax = linsolve(deltab); 
                    //swLinSolve.Stop();
                    for (int i = 0; i < N; i++) resp[i] -= deltax[i];
                }
            }

            //swResto.Stop();


            return resp;
        }

        private float[] linsolveCL(float[] bb)
        {
            float[] resp = new float[N];

            CLb.WriteToDevice(bb);

            CLCalc.Program.Variable[] args = new CLCalc.Program.Variable[] { CLcholDec, CLy, CLb, CLoffSet };

            int[] offset = new int[1];

            //Forward substitution
            for (int i = 0; i < N; i++)
            {
                offset[0] = i;
                CLoffSet.WriteToDevice(offset);
                kernelComputeTerm.Execute(args, 1);

                kernelUpdateBForward.Execute(args, N - i - 1);
            }

            //Backward subst. Stores answer in CLb
            args = new CLCalc.Program.Variable[] { CLcholDec, CLb, CLy, CLoffSet };
            //Backward substitution
            for (int i = N - 1; i >= 0; i--)
            {
                offset[0] = i;
                CLoffSet.WriteToDevice(offset);
                kernelComputeTerm.Execute(args, 1);

                kernelUpdateYBackward.Execute(args, i);
            }

            CLb.ReadFromDeviceTo(resp);
            return resp;
        }

        /// <summary>Solves system Ax = b and returns x</summary>
        /// <param name="bb">b vector</param>
        private float[] linsolve(float[] bb)
        {
            float[] b = (float[])bb.Clone();
            float[] y = new float[N];
            float[] resp = new float[N];

            //Forward substitution
            for (int i = 0; i < N; i++)
            {
                y[i] = b[i] / cholDec[((i * (i + 1)) >> 1) + i];

                for (int j = i + 1; j < N; j++)
                {
                    b[j] -= cholDec[((j * (j + 1)) >> 1) + i] * y[i];
                }
            }

            //Backward substitution
            for (int i = N - 1; i >= 0; i--)
            {
                resp[i] = y[i] / cholDec[((i * (i + 1)) >> 1) + i];

                for (int j = 0; j < i; j++)
                {
                    y[j] -= cholDec[((i * (i + 1)) >> 1) + j] * resp[i];
                }
            }

            return resp;
        }

        /// <summary>Retrieves the Determinant of this matrix</summary>
        public float Determinant
        {
            get
            {
                if (!IsCholeskyFactorized) this.ComputeCholesky();

                float det = 1;
                for (int i = 0; i < N; i++)
                {
                    det *= cholDec[((i * (i + 1)) >> 1) + i];
                }
                return det * det;
            }
        }
        #endregion

        #region Computation of eigenvalues

        /// <summary>Uses power iteration to compute eigenvalues of a matrix. The method looses accuracy for smaller eigenvectors</summary>
        /// <param name="qtdValues">Number of eigenvalues to compute</param>
        /// <param name="Eigenvectors">Associated eigenvectors</param>
        public List<float> Eigenvalues(int qtdValues, out List<float[]> Eigenvectors)
        {
            if (qtdValues > this.N) qtdValues = N;

            Eigenvectors = new List<float[]>();
            List<float> EigVals = new List<float>();

            int count = 0;

            for (int curEigIdx = 0; curEigIdx < qtdValues; curEigIdx++)
            {
                //Computes eigenvalue
                float eigAnt = 0, eig = 1;
                Random rnd = new Random();
                float[] b = new float[this.N];
                for (int k = 0; k < b.Length; k++) b[k] = (float)rnd.NextDouble();

                while (Math.Abs((eigAnt - eig) / eig) > 5e-7f || Math.Abs(eig) < 5E-7f)
                {
                    b = this * b;
                    eigAnt = eig;
                    eig = (float)Math.Sqrt(Dot(b, b));

                    float temp = 1.0f / eig;
                    for (int k = 0; k < b.Length; k++) b[k] *= temp;

                    count++;
                }

                if (EigVals.Count == 0 || eig <= EigVals[EigVals.Count - 1])
                {
                    Eigenvectors.Add(b);
                    EigVals.Add(eig);
                }
                else
                {
                    //numerical instability
                    curEigIdx = qtdValues;
                }


                //Computes new matrix M - lambda*eN*transpose(eN)
                for (int i = 0; i < N; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        this[i, j] -= eig * b[i] * b[j];
                    }
                }
            }

            return EigVals;
        }

        #endregion

        #region Operations (product, etc)
        /// <summary>Matrix vector multiplication</summary>
        public static float[] operator *(floatSymPosDefMatrix A, float[] v)
        {
            if (v.Length != A.N) throw new Exception("Incompatibe dimensions for matrix - vector product");

            if (A.N <= 200) return A.MultiplyNoCL(v);
            else return A.MultiplyCL(v);
        }
        /// <summary>Matrix vector multiplication</summary>
        public static float[] operator *(float[] v, floatSymPosDefMatrix A)
        {
            if (v.Length != A.N) throw new Exception("Incompatibe dimensions for matrix - vector product");

            if (A.N <= 200) return A.MultiplyNoCL(v);
            else return A.MultiplyCL(v);
        }

        /// <summary>Dot product</summary>
        public static float Dot(float[] a, float[] b)
        {
            float val = 0;
            if (a.Length != b.Length) throw new Exception("Incompatibe dimensions for inner product");

            int n = a.Length;

            for (int i = 0; i < n; i++) val += a[i] * b[i];

            return val;
        }

        private float[] MultiplyNoCL(float[] v)
        {
            float[] resp = new float[N];

            for (int i = 0; i < N; i++)
            {
                float val = 0;
                for (int k = 0; k < N; k++)
                {
                    val += this[i, k] * v[k];
                }
                resp[i] = val;
            }
            return resp;
        }
 
        private float[] MultiplyCL(float[] v)
        {
            if (!IsMatrixInClMemoryUpdated)
            {
                CLthisCp.WriteToDevice(this.values);
                IsMatrixInClMemoryUpdated = true;
            }
            CLb.WriteToDevice(v);

            kernelMatrVecMultiply.Execute(new CLCalc.Program.Variable[] { CLthisCp, CLb, CLy }, N);

            float[] resp = new float[N];
            CLy.ReadFromDeviceTo(resp);

            return resp;
        }

        #endregion

        #region Preconstructed matrices

        /// <summary>Returns the identity matrix</summary>
        /// <param name="n">Matrix dimension nxn</param>
        public static floatSymPosDefMatrix Identity(int n)
        {
            floatSymPosDefMatrix M = new floatSymPosDefMatrix(n);

            for (int i = 0; i < n; i++) M[i, i] = 1;

            return M;
        }

        #endregion

        #region Useful functions and nonlinear LS


        #region Computation of AtA and Atb
        /// <summary>Computes transpose(A)*A</summary>
        /// <param name="A">Original matrix</param>
        public static floatSymPosDefMatrix AuxLeastSquaresAtA(float[,] A)
        {
            return AuxLeastSquaresAtA(A, null);
        }

        /// <summary>Computes transpose(A)*A using weights W</summary>
        /// <param name="A">Original matrix</param>
        /// <param name="W">Measurement weight vector</param>
        public static floatSymPosDefMatrix AuxLeastSquaresAtA(float[,] A, float[] W)
        {
            int m = A.GetLength(0);
            int n = A.GetLength(1);

            if (W != null && W.Length != m) throw new Exception("Incompatible Weight dimensions");

            if (m > 400 && n > 50 && CLCalc.CLAcceleration == CLCalc.CLAccelerationType.UsingCL)
            {
                return AuxLSAtACL(A, W);
            }
            else
            {
                return AuxLeastSquaresAtAnoCL(A, W);
            }
        }

        /// <summary>Computes transpose(A)*A and transpose(A)*b weighted by W</summary>
        /// <param name="A">Original matrix</param>
        /// <param name="W">Measurement weight vector</param>
        private static floatSymPosDefMatrix AuxLeastSquaresAtAnoCL(float[,] A, float[] W)
        {
            //A (mxn), AtA (nxn) positive semidef symmetric
            int m = A.GetLength(0);
            int n = A.GetLength(1);

            float[] AtAvals = new float[(n * (n + 1)) >> 1];

            if (W != null)
            {
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        double val = 0;
                        for (int k = 0; k < m; k++)
                        {
                            val += A[k, i] * A[k, j] * W[k];
                        }
                        AtAvals[((i * (i + 1)) >> 1) + j] = (float)val;
                    }
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    for (int j = 0; j <= i; j++)
                    {
                        double val = 0;
                        for (int k = 0; k < m; k++)
                        {
                            val += A[k, i] * A[k, j];
                        }
                        AtAvals[((i * (i + 1)) >> 1) + j] = (float)val;
                    }
                }
            }

            return new floatSymPosDefMatrix(AtAvals);
        }

        /// <summary>Computes transpose(A)*A and transpose(A)*b weighted by W using OpenCL</summary>
        private static floatSymPosDefMatrix AuxLSAtACL(float[,] A, float[] W)
        {
            int m = A.GetLength(0);
            int n = A.GetLength(1);

            if (W == null)
            {
                W = new float[m];
                for (int i = 0; i < m; i++) W[i] = 1;
            }

            float[] Avec = new float[m * n];
            float[] AtA = new float[(n * (n + 1)) >> 1];
            int[] dims = new int[] { m, n };

            for (int i = 0; i < m; i++)
                for (int j = 0; j < n; j++)
                    Avec[j + n * i] = A[i, j];

            CLCalc.Program.Variable CLA = new CLCalc.Program.Variable(Avec);
            CLCalc.Program.Variable CLW = new CLCalc.Program.Variable(W);
            CLCalc.Program.Variable CLAtA = new CLCalc.Program.Variable(AtA);
            CLCalc.Program.Variable CLdims = new CLCalc.Program.Variable(dims);

            CLCalc.Program.Variable[] args = new CLCalc.Program.Variable[] { CLA, CLW, CLAtA, CLdims };
            kernelComputeAtWA.Execute(args, AtA.Length);

            CLAtA.ReadFromDeviceTo(AtA);

            return new floatSymPosDefMatrix(AtA);
        }

        /// <summary>Computes transpose(A)*b weighted by W</summary>
        /// <param name="A">Original matrix</param>
        /// <param name="b">Vector to multiply</param>
        private static float[] AuxLeastSquaresAtb(float[,] A, float[] b)
        {
            int m = A.GetLength(0);
            float[] W = new float[m];
            for (int i = 0; i < W.Length; i++) W[i] = 1;

            return AuxLeastSquaresAtb(A, b, W);
        }

        /// <summary>Computes transpose(A)*b</summary>
        /// <param name="A">Original matrix</param>
        /// <param name="b">Vector to multiply</param>
        /// <param name="W">Measurement weight vector</param>
        public static float[] AuxLeastSquaresAtb(float[,] A, float[] b, float[] W)
        {
            int m = A.GetLength(0);
            int n = A.GetLength(1);

            float[] resp = new float[n];

            if (b.Length != m) throw new Exception("Incompatible dimensions");
            if (W.Length != m) throw new Exception("Incompatible Weight dimensions");

            for (int i = 0; i < n; i++)
            {
                double val = 0;
                for (int k = 0; k < m; k++)
                {
                    val += A[k, i] * b[k] * W[k];
                }
                resp[i] = (float)val;
            }

            return resp;
        }
        #endregion

        #region Solution of ordinary least-squares problems

        /// <summary>Computes least squares fitting of Ax = b weighted by W and returns x</summary>
        public static float[] LeastSquares(float[,] A, float[] b, float[] W)
        {
            floatSymPosDefMatrix AtA = floatSymPosDefMatrix.AuxLeastSquaresAtA(A, W);
            float[] Atb = floatSymPosDefMatrix.AuxLeastSquaresAtb(A, b, W);

            return AtA.LinearSolve(Atb, true);
        }
        /// <summary>Computes least squares fitting of Ax = b and returns x</summary>
        public static float[] LeastSquares(float[,] A, float[] b)
        {
            floatSymPosDefMatrix AtA = floatSymPosDefMatrix.AuxLeastSquaresAtA(A);
            float[] Atb = floatSymPosDefMatrix.AuxLeastSquaresAtb(A, b);

            return AtA.LinearSolve(Atb, true);
        }

        #endregion

        #region Nonlinear least squares, http://www.alkires.com/103/chap6.pdf

        /// <summary>Delegate to compute residues and gradients based on current estimate x [n]. Returns residues r [m] and gradients gradR [m , n], j-th component
        /// of gradient of residue r[i] = [i,j] = gradR[i,j] </summary>
        /// <param name="x">Current estimate</param>
        /// <param name="r">Residues</param>
        /// <param name="gradR">Gradient of residue functions</param>
        /// <param name="ComputeGrads">Is the method requiring computation of gradients now?</param>
        public delegate void ComputeResidueGrad(float[] x, ref float[] r, ref float[,] gradR, bool ComputeGrads);

        /// <summary>Computes nonlinear least squares using user functions to evaluate residues and their gradients</summary>
        /// <param name="f">Function that computes residues [m] and their gradients [grad r1; grad r2] m x n (each gradient in one line) [i,j] = gradR[i,j]</param>
        /// <param name="x">Intial guess</param>
        /// <param name="m">Number of residue equations</param>
        /// <param name="maxiter">Maximum number of iterations</param>
        /// <param name="err">Adjustment error</param>
        /// <param name="eps">Convergence parameter</param>
        public static float[] NonLinearLS(ComputeResidueGrad f, float[] x, int m, int maxiter, ref double err, float eps)
        {
            int n = x.Length;
            float alpha = 0.002f;

            float[,] A = new float[m, n];
            float[] r = new float[m];

            double errAnt = 0;

            for (int i = 0; i < maxiter; i++)
            {
                //Computes residues and gradient
                f(x, ref r, ref A, true);

                errAnt = err;
                err = NormAtb(A, r, m, n);

                //if (errAnt == err) it means algorithm is not converging at all
                if (err < eps || errAnt == err || double.IsNaN(err)) i = maxiter;
                else
                {
                    floatSymPosDefMatrix AtA = floatSymPosDefMatrix.AuxLeastSquaresAtA(A);
                    float[] v = AtA.LinearSolve(floatSymPosDefMatrix.AuxLeastSquaresAtb(A, r), false);

                    for (int k = 0; k < v.Length; k++) v[k] = -v[k];

                    //Line search

                    //||r||²
                    float normRSquared = 0;
                    for (int k = 0; k < r.Length; k++) normRSquared += r[k] * r[k];

                    //2transpose(r)Av
                    float transpRAv = 0;
                    for (int p = 0; p < m; p++)
                    {
                        float val = 0;
                        for (int q = 0; q < n; q++) val += A[p, q] * v[q];
                        transpRAv += r[p] * val;
                    }
                    transpRAv *= 2.0f;

                    float t = 2.0f;
                    //iterates while sum(ri*(x+tv)^2)>||r||²+alpha*2*transpose(r)*A*v*t
                    float lhs = 1;
                    float rhs = 0;

                    float[] newX = (float[])x.Clone();

                    while (lhs > rhs)
                    {
                        t *= 0.5f;

                        //Update x
                        for (int k = 0; k < x.Length; k++) newX[k] = x[k] + v[k] * t;

                        //Update r
                        f(newX, ref r, ref A, false);

                        lhs = 0;
                        for (int k = 0; k < m; k++) lhs += r[k] * r[k];
                        rhs = normRSquared + alpha * transpRAv * t;
                    }

                    //if (!float.IsNaN(rhs)) 
                        x = newX;
                }
            }

            return x;
        }

        private static double NormAtb(float[,] A, float[] r, int m, int n)
        {
            double resp = 0;
            for (int i = 0; i < n; i++)
            {
                double val = 0;
                for (int k = 0; k < m; k++)
                {
                    val += A[k, i] * r[k];
                }
                resp += val * val;
            }
            resp = Math.Sqrt(resp);
            return resp;
        }


        #endregion

        #endregion

        #region Nonlinear optimization problem constructor
        /// <summary>Nonlinear optimization problem</summary>
        public class NonlinearOptimizProblem
        {
            /// <summary>Generic residue function of type F(x) - y</summary>
            public abstract class ResidueFunction
            {
                /// <summary>Value that function should have</summary>
                public float y;
                /// <summary>Local parameters of this function</summary>
                public float[] X;
                /// <summary>Indexes of local parameters in global unknown vector</summary>
                public int[] GlobalIndex;

                /// <summary>Computes residue and gradient of this Residue function</summary>
                /// <param name="x">Global optimization variable</param>
                /// <param name="ComputeGrad">Compute gradient?</param>
                /// <param name="Gradient">Global gradient of this function</param>
                public float ComputeResidueGradient(float[] x, bool ComputeGrad, out float[] Gradient)
                {
                    for (int i = 0; i < X.Length; i++)
                    {
                        if (GlobalIndex[i] >= 0) X[i] = x[GlobalIndex[i]];
                    }

                    float res; float[] LocalGrad;
                    if (ComputeGrad)
                    {
                        res = F(true, out LocalGrad) - this.y;
                        Gradient = new float[x.Length];

                        for (int i = 0; i < X.Length; i++)
                        {
                            if (GlobalIndex[i] >= 0) Gradient[GlobalIndex[i]] = LocalGrad[i];
                        }

                    }
                    else
                    {
                        res = F(false, out LocalGrad) - this.y;
                        Gradient = null;
                    }


                    return res;
                }

                /// <summary>For simulation purposes, compute local Y as a function of a global optimization variable</summary>
                /// <param name="x">Optimization variable</param>
                public void InitY(float[] x)
                {
                    this.y = 0;
                    float[] G;
                    this.y = ComputeResidueGradient(x, false, out G);
                }

                /// <summary>Computes function value and gradient using local information</summary>
                /// <param name="ComputeGrad">Compute gradient?</param>
                /// <param name="Gradient">Local gradient output</param>
                public abstract float F(bool ComputeGrad, out float[] Gradient);
            }

            /// <summary>List of residue functions of this nonlinear optimization problem</summary>
            public List<ResidueFunction> ResidueFunctions = new List<ResidueFunction>();

            /// <summary>Computes residues and gradients of residue functions of this nonlinear optimization problem</summary>
            /// <param name="x">Global optimization vector</param>
            /// <param name="r">Residues</param>
            /// <param name="gradR">Gradient</param>
            /// <param name="ComputeGrads">Compute gradients?</param>
            public void ComputeResidueGrad(float[] x, ref float[] r, ref float[,] gradR, bool ComputeGrads)
            {
                if (!ComputeGrads)
                {
                    float[] Grad;
                    for (int i = 0; i < ResidueFunctions.Count; i++)
                    {
                        r[i] = ResidueFunctions[i].ComputeResidueGradient(x, false, out Grad);
                    }
                }
                else
                {
                    float[] Grad;
                    for (int i = 0; i < ResidueFunctions.Count; i++)
                    {
                        r[i] = ResidueFunctions[i].ComputeResidueGradient(x, true, out Grad);
                        for (int j = 0; j < ResidueFunctions[i].GlobalIndex.Length; j++)
                        {
                            if (ResidueFunctions[i].GlobalIndex[j] >= 0) gradR[i, ResidueFunctions[i].GlobalIndex[j]] = Grad[ResidueFunctions[i].GlobalIndex[j]];
                        }
                    }

                }
            }

            /// <summary>Solves this nonlinear optimization problem</summary>
            /// <param name="x">Global optimization vector, initial guess</param>
            public float[] Solve(float[] x)
            {
                double err = 0;
                return floatSymPosDefMatrix.NonLinearLS(this.ComputeResidueGrad, x, ResidueFunctions.Count, 1000, ref err, 5e-5f * 0.5f);
            }

            /// <summary>Solves this nonlinear optimization problem</summary>
            /// <param name="x">Global optimization vector, initial guess</param>
            /// <param name="MAXITER">Maximum number of iterations</param>
            /// <param name="tol">Convergence tolerance. Default is 2.5E-6f</param>
            public float[] Solve(float[] x, int MAXITER, float tol)
            {
                double err = 0;
                return floatSymPosDefMatrix.NonLinearLS(this.ComputeResidueGrad, x, ResidueFunctions.Count, MAXITER, ref err, tol);
            }

            /// <summary>Example residue functions</summary>
            public static class SampleResidueFunctions
            {
                /// <summary>Exponential residue</summary>
                public class ResidueExp : ResidueFunction
                {
                    /// <summary>Exponential residue example constructor</summary>
                    /// <param name="T">Multiply factor</param>
                    public ResidueExp(float T)
                    {
                        this.X = new float[2];
                        this.GlobalIndex = new int[2];
                        this.t = T;
                    }

                    /// <summary>Multiply factor</summary>
                    public float t = 1.0f;

                    /// <summary>Example function</summary>
                    /// <param name="ComputeGrad">Compute gradient?</param>
                    /// <param name="Gradient">Gradient</param>
                    public override float F(bool ComputeGrad, out float[] Gradient)
                    {
                        float temp = t * X[0] * X[1] + X[1] * (float)Math.Exp(t * 0.1);
                        float resp = temp;

                        if (ComputeGrad)
                        {
                            Gradient = new float[2];
                            Gradient[0] = t * X[1];
                            Gradient[1] = (float)Math.Exp(t * 0.1) + t * X[0];
                        }
                        else Gradient = null;

                        return resp;
                    }
                }
            }
        }
        #endregion
    }
}
