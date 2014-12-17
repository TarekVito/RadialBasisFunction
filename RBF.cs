using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace NN_Project
{
    class RBF
    {
        int numRBFUnits,maxIters;
        double sigmaSQ,ETA,MSEMIN;
        public Double pWeights;
        List<Double> dataCenters;
        public RBF(int _numRBFUnits, int _maxIters, double _ETA, double _MSEMIN)
        {
            ETA = _ETA;
            MSEMIN = _MSEMIN;
            maxIters = _maxIters;
            numRBFUnits = _numRBFUnits;
        }
        Double getRBFs(Double data)
        {
            Double RBFs = new Double(numRBFUnits);
            for (int i = 0; i < numRBFUnits; ++i)
                RBFs[i] = calcPhi(data,dataCenters[i]);
            return RBFs;
        }
        double calcPhi(Double pattern, Double center)
        { 
            return Math.Exp(-getDist(pattern,center) / (2 * sigmaSQ));
        }
        double getDist(Double a, Double b)
        { 
            double dist=0;
            for (int i = 0; i < a.vec.Length; ++i)
            {
                if (double.IsNaN(dist + (a[i] - b[i]) * (a[i] - b[i])))
                    return -1;
                dist += (a[i] - b[i]) * (a[i] - b[i]);
            }
            return dist;
        }
        void calcSigma(List<List<double>> trainingData)
        {
            double maxVal = -1;
            for (int i = 0; i < dataCenters.Count; ++i)
                for (int j = i + 1; j < dataCenters.Count; ++j)
                    maxVal = Math.Max(maxVal, getDist(dataCenters[i], dataCenters[j]));
            sigmaSQ = maxVal / (2 * dataCenters.Count);
        }
        void calcCenters(List<List<double>> trainingData)
        {
            dataCenters = new List<Double>();
            List<List<double>> centers = new KMeansPP(numRBFUnits, trainingData).runKMean();
            for (int i = 0; i < centers.Count; ++i)
                dataCenters.Add(new Double(centers[i]));
        }
        public double startTraining(List<List<double>> trainingData, List<int> labels)
        {
            calcCenters(trainingData);
            calcSigma(trainingData);
            
            List<Double> RBFData = new List<Double>();
            for (int i = 0; i < trainingData.Count; ++i)
                RBFData.Add(getRBFs(new Double(trainingData[i])));

            Double initWeights = new Double(numRBFUnits);
            Random rand = new Random();
            for (int i = 0; i < initWeights.vec.Length; ++i)
                initWeights[i] = rand.NextDouble();

            double summ = 0, MES = 0;
            bool finished = false;
            for (int k = 0; k < maxIters && !finished; ++k)
            {
                summ = 0;
                for (int i = 0; i < RBFData.Count; ++i)
                {
                    double actVal = RBFData[i].mul(initWeights);
                    double diff = (labels[i] - actVal) / labels[i];
                    summ += (0.5 * diff * diff);
                    Double old = new Double(initWeights);
                    if (Math.Abs(diff) > 0.00001)
                        initWeights = new Double(initWeights.add(new Double(RBFData[i].mulCons(diff).mulCons(ETA))));
                }
                MES = (1 / (double)RBFData.Count) * summ;
                if (MES < MSEMIN)
                    finished = true;
            }

            pWeights = new Double(initWeights);

            return MES;
        }
        public double startTesting(List<List<double>> testingData, List<int> labels)
        {
            List<Double> RBFData = new List<Double>();
            for (int i = 0; i < testingData.Count; ++i)
                RBFData.Add(getRBFs(new Double(testingData[i])));

            double diffSum = 0;
            for (int i = 0; i < RBFData.Count; ++i)
            {
                double actVal = RBFData[i].mul(pWeights);
                double diff = (labels[i] - actVal) / labels[i];
                diffSum += diff * diff * 0.5;
            }
            double MES = (1 / (double)RBFData.Count) * diffSum;
            return MES;
        }
        public int getClass(List<double> p)
        {
            return (int)Math.Round(getRBFs(new Double(p)).mul(pWeights));
        }
    }
    public class Double
    {
        public double[] vec;
        public Double(int n)
        {
            vec = new double[n];
        }
        public double this[int i]
        {
            get { return vec[i]; }
            set { vec[i] = value; }
        }
        public Double(List<double> n)
        {
            vec = new double[n.Count];
            for (int i = 0; i < n.Count; ++i)
                vec[i] = n[i];
        }
        public Double(Double n) //Copy Constructor
        {
            vec = new double[n.vec.Length];
            for (int i = 0; i < n.vec.Length; ++i)
                vec[i] = n.vec[i];
        }
        public Double add(Double vec1)
        {
            Double res = new Double(vec.Length);
            for (int i = 0; i < vec1.vec.Length; ++i)
                res.vec[i] = vec[i] + vec1.vec[i];
            return res;
        }
        public Double mulCons(double num)
        {
            Double res = new Double(vec.Length);
            for (int i = 0; i < vec.Length; ++i)
                res.vec[i] = vec[i] * num;
            return res;
        }
        public double mul(Double vec1)
        {
            double sum = 0;
            for (int i = 0; i < vec1.vec.Length; ++i)
                sum += vec[i] * vec1.vec[i];
            return sum;
        }
    }


}
