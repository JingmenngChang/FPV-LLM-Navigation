## PCGWorker

------

基于WFC的 tile-map 生成系统

### 依赖项

* json
* opencv-python
* numpy
* Cython
* matplotlib
* colorama

### 运行过程

1. 编译 Cython 代码

   ```bash
   python3 setup.py build_ext --inplace
   ```

   

2. 运行示例程序

   ```bash
   python3 PCG_generate.py
   ```

   

### PCGWorker 基本过程

1. 基本生成：Generate

   从零开始生成一个符合规则的随机结构地图（seed），不包含任何优化和评估过程。

   ![](./code_structure-Page-4.png)

2. 变异：Mutate

   即条件式生成，给定一个初始地图（seed），对其变异生成新的地图（map）。

   ![](./code_structure-Page-5.png)

3. 优化：Optimize

   给定任一基本结构（seed），对其进行低熵优化，经过 N 个周期后形成原始结构的低熵演变版本（map）。

   ![](./code_structure-Page-6.png)

4. **以上述三个过程为基础，可以搭建出多种不同的生成过程，具体详见 Examples 部分**

### PCGWorker 功能介绍

系统中和地图相关的**数据部分**被封装于 Wave 类中，各个数据成员如下：

1. entropy ：熵值矩阵（计算中间结果）
2. oriented_weight_dict ：权重矩阵字典（计算中间结果）
3. **wave_oriented ：地图数据（输出结果）**
4. **mask ：连通域标注矩阵（输出结果）**

系统中所有基本功能被封装于 PCGWorker （PCG生成器）类中，其各个函数功能简介如下：

1. 生成器初始化：PCGWorker(self, width, height, num_processes = 0)

2. build_wave(self)

   * 生成原始计算缓存（尚未进行坍缩操作的 Wave 对象）

   * 返回值：Wave 原始地图

3. build_wave_conditional(self, wave_old = Wave(9,9),new_weight = 81)

   * 根据已经完成坍缩的地图 wave_old ，生成条件式计算缓存（尚未进行坍缩操作的 Wave 对象，但坍缩概率经过 wave_old 调制）
   * 返回值：Wave 原始地图

4. render(self, wave = Wave(9,9), wind_name = "canvas",write_ = False,write_id = 0,output = False, verbose = False, border = True)

   * 对已经坍缩完成的 wave 进行渲染
     * write_  文件写入开关
     * write_id 文件输出编号
     * output 返回图像开关
     * verbose 命令行输出开关
     * border 图像边界绘制开关
   * 返回值：opencv 图像（numpy数组）

5. super_render(self,wave1,wave2,wave3)

   * 将 wave1-wave3 三张地图拼接成一个画面输出
   * 返回值：opencv 图像（numpy数组）

6. **generate(self)**

   * 基本生成
   * 返回值：wave

7. **mutate(self, wave_old = Wave(9,9),weight_=81)**

   * 变异
   * 返回值：wave

8. **optimize(self, wave = Wave(9,9), max_iter = 500, fitness_report = True)**

   * 优化
   * 返回值：wave

9. **mutate_and_optimize(self, wave = Wave(9,9), max_iter = 500, fitness_report = True)**

   * 低熵变异（变异+优化）
   * 返回值：wave

10. **connectivity_analysis(self, wave = Wave(9,9),visualize_ = True, to_file = True, filename = "connectivity_map.json")**

    ![](./connectivity_map_mask0.bmp)

    ![](./1.png)

    ![](./2.png)

    ![](./3.png)

    

    1. 地图连通性分析
       * wave 要分析的（已经完成坍缩的）地图
       * visualize_ 可视化开关
       * to_file 输出到文件开关
       * filename 输出文件名（会同时输出 json 文件和 bmp 图像）
    2. 输出
       * 连通图 mask 文件 connectivity_map.json ，连通图矩阵的分辨率为 (self.Width * 20 , self.Height * 20) 像素
       * 彩色连通区域图像 connectivity_map.bmp
       * 返回值：
         * mask 连通图矩阵， (self.Width * 20 , self.Height * 20) int 类型
         * mask_visual 彩色连通区域图像

11. **文件操作**

    1. to_file(self, wave = Wave(9,9), filename = "wave.json") ：将坍缩完成后的地图写出到 json 文件
    2. from_file(self,filename = "wave.json") ：从 json 文件读入地图

### Examples

使用上述基本功能，可以组合出多种不同操作，每一种操作采用一个示例程序进行演示：

1. PCG_generate.py

   1. 直接生成随机地图，并且对生成结果进行连通性分析

   ![](./code_structure-Page-7.png)

2. PCG_mutate.py

   1. 先生成一个随机地图，然后根据这个地图进行一步变异

   ![](./code_structure-Page-8.png)

3. PCG_mutate_continous.py

   1. 先生成一个随机地图，然后从这个地图开始，连续变异

4. PCG_mutate_from_json.py

   1. 从 json 文件中读入一个地图，然后对这个地图进行变异

5. PCG_optimize.py

   1. 先生成一个随机地图，然后对这个地图进行优化

   ![](./code_structure-Page-9.png)

6. PCG_mutate_and_optimize.py

   1. 先生成一个随机地图，然后计算它的优化结构，然后以这个结构为种子，进行它的低熵结构演变

   ![](./code_structure-Page-10.png)



------

### TODO ( all done )

1. ~~一旦在每一次propagate之后更新全局熵值，就会让propagate过程变得很慢~~
2. ~~PCGWorker 有一定的生成失败概率，需要详细检查，否则会影响速度~~
3. ~~偶尔会 segment fault~~
4. ~~如果丢弃 update_wave_entropy 步骤，从肉眼上看效果可能不受影响，具体影响需要后续分析~~
   1. ~~TODO : 通过直方图分布对两种方法生成的结果进行比较~~
5. ~~propagate 中发现了一种极快的计算方法：每次仅对已完成坍缩的元素进行 propagate。但是这种方法生成的地图效果不好，缺乏层次感~~
6. ~~按照正统方法计算速度很慢，时间主要花在完全传播过程上，需要系统的分析不完全传播和完全传播的效果差异~~







