#ifndef TEST_DATA_H
#define TEST_DATA_H

// 3 * 4 * 4
static float testImage[] = {
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
    34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47
};


// 2 * 3 * 2 * 2
static float testKernelCol[] = {
    0, 12,
    1, 13,
    2, 14,
    3, 15,
    4, 16,
    5, 17,
    6, 18,
    7, 19,
    8, 20,
    9, 21,
    10, 22,
    11, 23
};

static float testKernelRow[] = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23
};

// 3 * 9 * 9
static float testData[] = {
    0.3869, 0.6784, 0.3602, 0.7166, 0.2600, 0.4458, 0.5419, 0.0131, 0.4740,
    0.0569, 0.8649, 0.1160, 0.8639, 0.5886, 0.9078, 0.6554, 0.7810, 0.1325,
    0.0750, 0.2943, 0.4093, 0.5352, 0.3288, 0.4674, 0.3874, 0.6407, 0.6977,
    0.2468, 0.6408, 0.4696, 0.0142, 0.1432, 0.4252, 0.6127, 0.7270, 0.7848,
    0.5873, 0.0176, 0.0656, 0.7689, 0.4545, 0.5342, 0.3070, 0.7558, 0.5052,
    0.7684, 0.3088, 0.0568, 0.5220, 0.4652, 0.1128, 0.7502, 0.6542, 0.3715,
    0.2043, 0.4806, 0.0830, 0.9004, 0.5042, 0.0231, 0.2202, 0.2384, 0.3946,
    0.1931, 0.4520, 0.5196, 0.4545, 0.2699, 0.9761, 0.5048, 0.0826, 0.9146,
    0.3818, 0.6928, 0.8899, 0.0933, 0.2275, 0.0271, 0.8812, 0.2876, 0.6365,
    0.5499, 0.9767, 0.5524, 0.6248, 0.3999, 0.6711, 0.5265, 0.5438, 0.3660,
    0.5799, 0.5642, 0.4833, 0.7769, 0.7269, 0.0984, 0.7524, 0.3820, 0.9593,
    0.4357, 0.1465, 0.1566, 0.2197, 0.8678, 0.1533, 0.7384, 0.6824, 0.3136,
    0.7447, 0.9706, 0.4157, 0.8670, 0.8803, 0.5360, 0.0622, 0.7319, 0.0095,
    0.4814, 0.4001, 0.5304, 0.0567, 0.7457, 0.8075, 0.7895, 0.0941, 0.6597,
    0.4792, 0.0737, 0.1505, 0.0827, 0.2700, 0.4155, 0.2973, 0.6922, 0.9371,
    0.1855, 0.2024, 0.4869, 0.7581, 0.0551, 0.7967, 0.0604, 0.5309, 0.8752,
    0.8919, 0.0767, 0.0496, 0.0121, 0.8442, 0.1641, 0.4343, 0.1822, 0.6477,
    0.8316, 0.1800, 0.5228, 0.3022, 0.9986, 0.7670, 0.1004, 0.2550, 0.3848,
    0.5481, 0.5200, 0.3472, 0.9197, 0.3221, 0.4623, 0.9296, 0.7910, 0.5925,
    0.4725, 0.9211, 0.3994, 0.1630, 0.0121, 0.7163, 0.1920, 0.7405, 0.9954,
    0.8662, 0.8741, 0.4116, 0.6710, 0.4931, 0.0043, 0.0653, 0.0274, 0.4146,
    0.2267, 0.3758, 0.7209, 0.2661, 0.1464, 0.8144, 0.1628, 0.7219, 0.0266,
    0.7632, 0.4512, 0.3775, 0.8247, 0.3419, 0.1181, 0.0808, 0.2637, 0.2596,
    0.3356, 0.3639, 0.3891, 0.1480, 0.2357, 0.5552, 0.5429, 0.7190, 0.2444,
    0.7246, 0.6382, 0.7154, 0.6352, 0.6540, 0.0176, 0.4514, 0.9858, 0.6169,
    0.1178, 0.6984, 0.3019, 0.6067, 0.2112, 0.4501, 0.9122, 0.7700, 0.9793,
    0.6556, 0.3319, 0.4913, 0.0886, 0.5591, 0.9598, 0.8932, 0.4202, 0.7483
};

// 3 * 3 * 5 * 5
static float test_kernel[] = {
    0.7077, 0.1797, 0.6213, 0.5535, 0.0906, 0.3716, 0.5622, 0.3558, 0.5898,
    0.7480, 0.1372, 0.3012, 0.8623, 0.7656, 0.7157, 0.4624, 0.6585, 0.8395,
    0.7153, 0.0498, 0.7933, 0.4696, 0.2986, 0.6118, 0.9517, 0.7678, 0.4261,
    0.9698, 0.9309, 0.1725, 0.1924, 0.4263, 0.9706, 0.6931, 0.5410, 0.4212,
    0.2222, 0.9750, 0.2099, 0.2472, 0.3375, 0.8229, 0.9612, 0.2929, 0.9996,
    0.7648, 0.6983, 0.9970, 0.2574, 0.1529, 0.3143, 0.6483, 0.1025, 0.6299,
    0.9449, 0.2738, 0.5596, 0.0497, 0.9011, 0.9561, 0.7859, 0.1281, 0.1263,
    0.1386, 0.7648, 0.5823, 0.0530, 0.3474, 0.5152, 0.7175, 0.2571, 0.7586,
    0.5180, 0.7360, 0.5649, 0.4856, 0.3317, 0.7638, 0.9364, 0.8168, 0.6731,
    0.9419, 0.0709, 0.3245, 0.7976, 0.8396, 0.8016, 0.0686, 0.7864, 0.4328,
    0.3010, 0.7515, 0.1429, 0.7183, 0.3661, 0.9491, 0.1808, 0.5935, 0.6380,
    0.3262, 0.6149, 0.7401, 0.5502, 0.2665, 0.3019, 0.0924, 0.9151, 0.8524,
    0.7022, 0.0251, 0.9143, 0.9624, 0.6386, 0.2608, 0.0534, 0.3201, 0.5009,
    0.7104, 0.1230, 0.4889, 0.8623, 0.9790, 0.0776, 0.2167, 0.8241, 0.8044,
    0.7650, 0.5747, 0.2749, 0.3564, 0.6872, 0.7488, 0.3535, 0.3526, 0.9506,
    0.4245, 0.1820, 0.5789, 0.2706, 0.0207, 0.6066, 0.7990, 0.3598, 0.3235,
    0.6316, 0.9885, 0.7180, 0.5005, 0.2757, 0.2062, 0.9315, 0.8075, 0.6852,
    0.7663, 0.2532, 0.6872, 0.7902, 0.9750, 0.9347, 0.3428, 0.4908, 0.5310,
    0.4350, 0.9633, 0.6539, 0.5312, 0.1677, 0.0230, 0.9854, 0.5069, 0.2364,
    0.5586, 0.8502, 0.0406, 0.5153, 0.9719, 0.3939, 0.2582, 0.1813, 0.1114,
    0.6273, 0.4462, 0.8342, 0.1779, 0.7128, 0.4279, 0.3488, 0.1341, 0.8128,
    0.6413, 0.1226, 0.1628, 0.3843, 0.7081, 0.6058, 0.9772, 0.8651, 0.4636,
    0.9399, 0.3048, 0.6593, 0.3182, 0.8277, 0.7719, 0.9680, 0.2513, 0.6648,
    0.3516, 0.3149, 0.0863, 0.9513, 0.7135, 0.6544, 0.2701, 0.3655, 0.5207,
    0.3854, 0.7289, 0.4418, 0.5238, 0.4428, 0.9536, 0.8000, 0.5085, 0.6912
};

#endif
