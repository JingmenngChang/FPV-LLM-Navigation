import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import xml.etree.ElementTree as ET
import os
from pathlib import Path

class Map3Dto2DConverter:
    def __init__(self, xml_file_path):
        """
        初始化转换器。
        - xml_file_path: 定义了瓦片（tile）信息的XML文件路径。
        """
        self.color_mapping = {
            'gray': '#8d999d',
            'blue': '#6bbefc',
            'yellow': '#bdcb41',
            'orange': '#f5bf87',
            'red': '#e67468',
            'white': '#f0ffff'
        }
        self.tile_mapping = self._parse_xml(xml_file_path)

    def _parse_xml(self, xml_file_path):
        """
        解析XML文件，为每个瓦片索引建立详细的映射。
        """
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        tile_mapping = {}
        tile_index = 0
        
        # 解析tiles
        for tile in root.find('tiles').findall('tile'):
            tile_name = tile.get('name')
            tile_type = self._get_tile_type(tile_name)
            colors = self._get_tile_colors(tile_name, tile_type)
            
            tile_mapping[tile_index] = {
                'name': tile_name,
                'type': tile_type,
                'colors': colors  # 存储颜色名称列表
            }
            tile_index += 1
            
        return tile_mapping

    def _get_tile_type(self, tile_name):
        """根据瓦片名称获取类型（cube, corner, ramp）。"""
        if tile_name.startswith('cube_'):
            return 'cube'
        elif tile_name.startswith('corner_'):
            return 'corner'
        elif tile_name.startswith('ramp_'):
            return 'ramp'
        return 'unknown'

    def _get_tile_colors(self, tile_name, tile_type):
        """
        从瓦片名称中提取颜色。
        - 'corner'类型需要两种颜色 (例如: corner_red_gray)。
        - 其他类型只需要一种颜色 (例如: ramp_blue)。
        """
        parts = tile_name.split('_')
        if tile_type == 'corner' and len(parts) >= 3:
            # 期望格式: corner_color1_color2
            return [parts[1], parts[2]]
        elif len(parts) >= 2:
            # 期望格式: type_color
            return [parts[1]]
        return ['gray'] # 默认颜色

    def _create_processed_grid(self, json_data):
        """
        根据JSON数据创建一个处理过的网格，包含每个位置的详细瓦片信息。
        这取代了原先只生成颜色网格的方法。
        """
        # 解析JSON数据
        if isinstance(json_data, str):
            data = json.loads(json_data)['data']
        else:
            data = json_data['data']
        
        grid_size = 9
        # 创建一个空网格，用于存放每个位置的瓦片信息字典
        processed_grid = [[None for _ in range(grid_size)] for _ in range(grid_size)]
        
        # 处理每个瓦片
        for i in range(0, len(data), 2):
            tile_id = data[i]
            rotation = data[i + 1]
            
            # 计算在网格中的位置
            tile_index = i // 2
            row = tile_index // grid_size
            col = tile_index % grid_size
            
            # 获取瓦片信息
            if tile_id in self.tile_mapping:
                tile_info = self.tile_mapping[tile_id]
                hex_colors = [self.color_mapping.get(c, '#808080') for c in tile_info['colors']]
                
                processed_grid[row][col] = {
                    'type': tile_info['type'],
                    'colors': hex_colors,
                    'rotation': rotation
                }
        
        return processed_grid

    def visualize_2d_map(self, processed_grid, save_path=None, ax=None, trap_positions=None, show_grid=False):
        """
        可视化2D地图。
        这个函数支持绘制'T'符号和复杂的'corner'方块。
        如果提供了ax，则在其上绘图；否则，创建新的fig和ax。
        """
        if ax is None:
            fig, ax_local = plt.subplots(1, 1, figsize=(10, 10))
        else:
            ax_local = ax
            fig = ax.get_figure()

        grid_size = len(processed_grid)
        # 将背景色设置为与灰色方块相同，以匹配示例图
        ax_local.set_facecolor(self.color_mapping['gray'])

        # 反转y轴，使(0,0)位于左上角，与数组索引匹配
        ax_local.set_ylim(grid_size, 0)
        ax_local.set_xlim(0, grid_size)

        for r in range(grid_size):
            for c in range(grid_size):
                tile = processed_grid[r][c]
                if tile is None:
                    continue

                x, y = c, r  # (x,y)是matplotlib patch的左下角坐标

                # --- 绘制 Corner 方块 ---
                if tile['type'] == 'corner':
                    color1, color2 = tile['colors']
                    rotation = tile['rotation']
                    
                    # 定义方块的四个顶点
                    bl, br, tr, tl = (x, y + 1), (x + 1, y + 1), (x + 1, y), (x, y)

                    # 根据旋转方向确定对角线和颜色分布
                    if rotation == 0: # 对角线 TL-BR, color1在右下, color2在左上
                        poly1 = patches.Polygon([tl, tr, br], fc=color2, ec='none')
                        poly2 = patches.Polygon([tl, bl, br], fc=color1, ec='none')
                    elif rotation == 1: # 对角线 BL-TR, color1在右上, color2在左下
                        poly1 = patches.Polygon([bl, tr, tl], fc=color2, ec='none')
                        poly2 = patches.Polygon([bl, tr, br], fc=color1, ec='none')
                    elif rotation == 2: # 对角线 TL-BR, 反色
                        poly1 = patches.Polygon([tl, tr, br], fc=color1, ec='none')
                        poly2 = patches.Polygon([tl, bl, br], fc=color2, ec='none')
                    else: # rotation == 3, 对角线 BL-TR, 反色
                        poly1 = patches.Polygon([bl, tr, tl], fc=color1, ec='none')
                        poly2 = patches.Polygon([bl, tr, br], fc=color2, ec='none')
                    
                    ax_local.add_patch(poly1)
                    ax_local.add_patch(poly2)

                # --- 绘制 Cube 和 Ramp 方块 ---
                else:
                    color = tile['colors'][0]
                    rect = patches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor='none')
                    ax_local.add_patch(rect)
                    
                    # --- 如果是 Ramp，添加 'T' 符号 ---
                    if tile['type'] == 'ramp':
                        # 游戏内的旋转值(0,1,2,3)映射到matplotlib的旋转角度(逆时针)
                        rotation_map = {0: 0, 1: 90, 2: 180, 3: 270}
                        angle = rotation_map.get(tile['rotation'], 0)
                        
                        ax_local.text(c + 0.5, r + 0.5, 'T',
                                ha='center', va='center', fontsize=24,
                                fontweight='bold', color='black',
                                rotation=angle)
        
        # Draw traps if provided
        if trap_positions:
            for r_trap, c_trap in trap_positions:
                rect = patches.Rectangle((c_trap, r_trap), 1, 1, linewidth=2, edgecolor='magenta', facecolor='none', linestyle='--', hatch='x')
                ax_local.add_patch(rect)

        # 添加白色网格线，以匹配示例图
        if show_grid:
            for i in range(grid_size + 1):
                ax_local.axhline(i, color='white', linewidth=2)
                ax_local.axvline(i, color='white', linewidth=2)
        
        ax_local.set_aspect('equal')
        ax_local.set_title('2D Top-down View', fontsize=16)
        
        # 移除坐标轴刻度
        ax_local.set_xticks([])
        ax_local.set_yticks([])
        
        # 只有在自己创建fig/ax时才进行保存或显示
        if ax is None:
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', dpi=150)
                plt.close(fig)
            else:
                plt.show()

    def batch_convert(self, input_folder, output_folder):
        """批量转换一个文件夹内的所有JSON地图。"""
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True, parents=True)
        
        # 修正了glob模式以正确查找.json文件
        json_files = sorted(list(input_path.glob("*.json")))
        
        if not json_files:
            print(f"Warning: No .json files found in {input_path}")
            return
            
        for i, json_file in enumerate(json_files):
            print(f"Processing {json_file.name} ({i+1}/{len(json_files)})")
            
            # 读取JSON文件并转换
            with open(json_file, 'r') as f:
                json_data = json.load(f)
            processed_grid = self._create_processed_grid(json_data)
            
            # 保存2D图像
            output_file = output_path / f"{json_file.stem}_2d.png"
            self.visualize_2d_map(processed_grid, save_path=output_file)

    def convert_single_file(self, json_file_path, output_path=None):
        """转换单个JSON文件并可视化。"""
        with open(json_file_path, 'r') as f:
            json_data = json.load(f)
        
        # 创建包含详细信息的处理网格
        processed_grid = self._create_processed_grid(json_data)
        
        # 可视化处理后的网格
        self.visualize_2d_map(processed_grid, save_path=output_path)
        
        return processed_grid
