"""
거름망 3D 형상 정의
- 외부: 돌기 구조, 나선형 구멍, 미세 슬롯
- 내부: 나선형 수로
"""

import numpy as np
import pyvista as pv
from typing import Dict, List


class RainFilterGeometry:
    """빗물 거름망 형상 클래스"""
    
    def __init__(self, params: Dict):
        """
        초기화
        
        Args:
            params: sim_params.json의 geometry 섹션
        """
        self.radius = params['filter_radius']
        self.height = params['filter_height']
        self.spiral_hole_dia = params['spiral_hole_diameter']
        self.spiral_hole_count = params['spiral_hole_count']
        self.slot_width = params['slot_width']
        self.slot_count = params['slot_count']
        self.protrusion_height = params['protrusion_height']
        self.protrusion_count = params['protrusion_count']
        self.channel_width = params['spiral_channel_width']
        self.channel_turns = params['spiral_channel_turns']
        
        # 형상 생성
        self.mesh = self._create_geometry()
    
    def _create_geometry(self) -> pv.MultiBlock:
        """전체 형상 생성"""
        geometry = pv.MultiBlock()
        
        # 1. 외부 실린더 (메인 바디)
        cylinder = pv.Cylinder(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            radius=self.radius,
            height=self.height
        )
        geometry.append(cylinder, "main_body")
        
        # 2. 돌기 구조 (상단)
        protrusions = self._create_protrusions()
        geometry.append(protrusions, "protrusions")
        
        # 3. 나선형 구멍 (측면)
        spiral_holes = self._create_spiral_holes()
        geometry.append(spiral_holes, "spiral_holes")
        
        # 4. 미세 슬롯 (하단)
        slots = self._create_slots()
        geometry.append(slots, "slots")
        
        # 5. 내부 나선형 수로
        spiral_channel = self._create_spiral_channel()
        geometry.append(spiral_channel, "spiral_channel")
        
        return geometry
    
    def _create_protrusions(self) -> pv.MultiBlock:
        """돌기 구조 생성 (상단 덮개 밀착 방지)"""
        protrusions = pv.MultiBlock()
        
        for i in range(self.protrusion_count):
            angle = 2 * np.pi * i / self.protrusion_count
            x = self.radius * 0.9 * np.cos(angle)
            y = self.radius * 0.9 * np.sin(angle)
            z = self.height / 2
            
            # 작은 원기둥으로 돌기 표현
            protrusion = pv.Cylinder(
                center=(x, y, z),
                direction=(0, 0, 1),
                radius=self.radius * 0.05,
                height=self.protrusion_height
            )
            protrusions.append(protrusion)
        
        return protrusions
    
    def _create_spiral_holes(self) -> pv.MultiBlock:
        """나선형 구멍 생성 (20-30mm 큰 이물질 차단)"""
        holes = pv.MultiBlock()
        
        for i in range(self.spiral_hole_count):
            # 나선형 배치
            t = i / self.spiral_hole_count
            angle = 4 * np.pi * t  # 2회전
            z = self.height * (0.3 + 0.4 * t)  # 중간 높이
            
            x = self.radius * np.cos(angle)
            y = self.radius * np.sin(angle)
            
            # 구멍 (실린더로 표현)
            hole = pv.Cylinder(
                center=(x, y, z),
                direction=(np.cos(angle), np.sin(angle), 0),
                radius=self.spiral_hole_dia / 2,
                height=self.radius * 0.2
            )
            holes.append(hole)
        
        return holes
    
    def _create_slots(self) -> pv.MultiBlock:
        """미세 슬롯 생성 (5mm 폭, 모래/분진 유입)"""
        slots = pv.MultiBlock()
        
        for i in range(self.slot_count):
            angle = 2 * np.pi * i / self.slot_count
            x = self.radius * 0.95 * np.cos(angle)
            y = self.radius * 0.95 * np.sin(angle)
            z = -self.height / 2 + 0.05
            
            # 얇은 직육면체로 슬롯 표현
            slot = pv.Box(
                bounds=(
                    x - self.slot_width / 2, x + self.slot_width / 2,
                    y - self.radius * 0.1, y + self.radius * 0.1,
                    z, z + 0.03
                )
            )
            slots.append(slot)
        
        return slots
    
    def _create_spiral_channel(self) -> pv.PolyData:
        """내부 나선형 수로 생성 (와류 형성)"""
        # 나선형 경로 생성
        theta = np.linspace(0, 2 * np.pi * self.channel_turns, 200)
        z = np.linspace(self.height / 2, -self.height / 2, 200)
        
        # 나선형 좌표
        r = self.radius * 0.7  # 내벽에서 약간 안쪽
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        
        # 경로를 따라 튜브 생성
        points = np.column_stack([x, y, z])
        spline = pv.Spline(points, 200)
        channel = spline.tube(radius=self.channel_width / 2)
        
        return channel
    
    def get_collision_surfaces(self) -> List[Dict]:
        """충돌 감지용 표면 정보 반환"""
        surfaces = []
        
        # 메인 실린더 표면
        surfaces.append({
            'type': 'cylinder',
            'radius': self.radius,
            'height': self.height,
            'center': np.array([0, 0, 0])
        })
        
        # 나선형 수로 표면 (와류 유도)
        surfaces.append({
            'type': 'spiral_channel',
            'width': self.channel_width,
            'turns': self.channel_turns
        })
        
        return surfaces
    
    def visualize(self, plotter: pv.Plotter = None, opacity: float = 0.3):
        """형상 시각화"""
        if plotter is None:
            plotter = pv.Plotter()
        
        # 메인 바디
        plotter.add_mesh(
            self.mesh["main_body"],
            color='lightgray',
            opacity=opacity,
            label='Main Body'
        )
        
        # 돌기
        plotter.add_mesh(
            self.mesh["protrusions"],
            color='orange',
            opacity=opacity + 0.2,
            label='Protrusions'
        )
        
        # 나선형 구멍
        plotter.add_mesh(
            self.mesh["spiral_holes"],
            color='red',
            opacity=opacity + 0.3,
            label='Spiral Holes'
        )
        
        # 슬롯
        plotter.add_mesh(
            self.mesh["slots"],
            color='blue',
            opacity=opacity + 0.3,
            label='Slots'
        )
        
        # 나선형 수로
        plotter.add_mesh(
            self.mesh["spiral_channel"],
            color='cyan',
            opacity=opacity + 0.4,
            label='Spiral Channel'
        )
        
        plotter.show()
        
        return plotter

# 테스트 코드
if __name__ == "__main__":
    import json
    import os
    
    # 설정 파일 경로 (src에서 실행 시)
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "sim_params.json")
    
    # 설정 로드
    with open(config_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    
    # 형상 생성
    print("거름망 형상 생성 중...")
    geometry = RainFilterGeometry(params['geometry'])
    
    # 오프스크린 렌더링 (SSH 환경에서 이미지 저장)
    print("오프스크린 렌더링 시작...")
    plotter = pv.Plotter(off_screen=True, window_size=[1920, 1080])
    
    # 메인 바디
    plotter.add_mesh(
        geometry.mesh["main_body"],
        color='lightgray',
        opacity=0.3,
        label='Main Body'
    )
    
    # 돌기
    plotter.add_mesh(
        geometry.mesh["protrusions"],
        color='orange',
        opacity=0.5,
        label='Protrusions'
    )
    
    # 나선형 구멍
    plotter.add_mesh(
        geometry.mesh["spiral_holes"],
        color='red',
        opacity=0.6,
        label='Spiral Holes'
    )
    
    # 슬롯
    plotter.add_mesh(
        geometry.mesh["slots"],
        color='blue',
        opacity=0.6,
        label='Slots'
    )
    
    # 나선형 수로
    plotter.add_mesh(
        geometry.mesh["spiral_channel"],
        color='cyan',
        opacity=0.7,
        label='Spiral Channel'
    )
    
    # 카메라 위치 설정
    plotter.camera_position = [(0.5, 0.5, 0.4), (0, 0, 0), (0, 0, 1)]
    plotter.add_axes()
    
    # 출력 디렉토리 생성
    output_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(output_dir, exist_ok=True)
    
    # 여러 각도에서 이미지 저장
    angles = [
        ("front", [(0.5, 0, 0.2), (0, 0, 0), (0, 0, 1)]),
        ("side", [(0, 0.5, 0.2), (0, 0, 0), (0, 0, 1)]),
        ("top", [(0, 0, 0.6), (0, 0, 0), (0, 1, 0)]),
        ("iso", [(0.4, 0.4, 0.4), (0, 0, 0), (0, 0, 1)])
    ]
    
    for view_name, camera_pos in angles:
        plotter.camera_position = camera_pos
        output_path = os.path.join(output_dir, f"geometry_{view_name}.png")
        plotter.screenshot(output_path)
        print(f"저장 완료: {output_path}")
    
    print("\n모든 이미지가 outputs 폴더에 저장되었습니다.")
    print("다음 명령어로 확인하세요:")
    print("  Get-ChildItem outputs")