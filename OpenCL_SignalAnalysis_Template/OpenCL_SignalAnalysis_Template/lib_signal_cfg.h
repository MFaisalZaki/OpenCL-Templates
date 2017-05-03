#ifndef _LIB_SIGNAL_CFG_H_
#define _LIB_SIGNAL_CFG_H_

#define SIGNAL_1D_DCT  0
#define SIGNAL_1D_IDCT 1
#define SIGNAL_2D_DCT  2
#define SIGNAL_2D_IDCT 3

#define SIGNAL_KERNEL_FILE_NAME "/Users/marwanfaisal/Documents/Desktop_Developer/OpenCL_SignalAnalysis_Template/OpenCL_SignalAnalysis_Template/Kernel_DCT.cl"

#define KERNEL_PRG_CNT 4
#define SIGNAL_KERNEL_LIST_NAMES {"computeDCT1D", "computeIDCT1D", "computeDCT2D", "computeIDCT2D"}

#endif /* _LIB_SIGNAL_CFG_H_ */
