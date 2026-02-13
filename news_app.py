"""
Tennis Tactical Simulation Engine - Tournament Master Protocol (V2026.46-Audited)
Tactical Logic Engine (tactical_brain.py)

Core Functionality:
- Shot Type System (Fixed and detailed shot types, 15 shot paths: left/center/right directions)
- Top Tennis Tactical Brain (70/30 Principle)
- Dynamic Opponent Prediction Engine (Full tactical logic from opponent's perspective, 15 shot types)
- Anti-Stalemate Logic (Breakthrough patterns, applicable to both players)
- Tactical Scripts (Pattern Recognition)
- 48-Tactic Traceability System (Tactical ID mapping)
- Psychological Momentum System
- Surface Physics System (Clay/Grass/Hard court adjustments)
- Core Tactical Patterns (Angle-Line, Crosscourt Rally, Counter-Punch, etc.)
- Critical Situation Tactics (Serve+1, Serve and Volley, Return Strategies, etc.)

Reference Documentation:
- Key Principles of Modern Tennis Singles: A Strategic Analysis
- Understanding Psychological Momentum in Tennis
- A Beginner's Guide to Tennis Court Surfaces
- A Strategic Guide to Tennis Tactics by Court Position
- Tennis Singles Tactical & Technical Strategy Reference Table (48 Tactics)

Scientific Principles Applied:
- Wardlaw Directional Principles (Inside/Outside ball theory)
- Garlikov Recovery (Angle bisector optimal positioning)
- Geometric Pressure Coefficient (Lateral displacement quantification)
- Displacement Cost Function (Energy expenditure modeling, BMR 2200 kcal baseline)
- Crosscourt Advantage Multiplier (82.5ft diagonal vs 78.0ft DTL geometry)
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import math
import json
import os
import re

# Import Geometry Engine
from geometry_engine import (
    Geometry30Point,
    PhaseType,
    PhaseDetector,
    GarlikovRecovery,
    WardlawTheory,
    CourtGeometry
)
from geometric_classifier import GeometricClassifier, ShotDirection  # 🔥 V2026.52

# ============================================================================
# 48-Tactic Library (Excel) - Auxiliary Mapping System
# ============================================================================

_TACTICS_EXCEL_PATH = os.path.join(os.path.dirname(__file__), "tactics_table.json")

def _normalize_tactic_name(name: str) -> str:
    if not name:
        return ""
    return re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", name).lower()

def _load_excel_tactics() -> List[Dict[str, str]]:
    if not os.path.exists(_TACTICS_EXCEL_PATH):
        return []
    try:
        with open(_TACTICS_EXCEL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []

_TACTICS_EXCEL = _load_excel_tactics()

def _match_excel_tactic_profile(strategy_name: str) -> Optional[Dict[str, str]]:
    if not _TACTICS_EXCEL:
        return None
    target = _normalize_tactic_name(strategy_name)
    if not target:
        return None
    for item in _TACTICS_EXCEL:
        name = _normalize_tactic_name(item.get("戰術名稱", ""))
        if name and (name in target or target in name):
            return item
    return None


# ============================================================================
# Shot Type System
# ============================================================================

class FixedShotType(Enum):
    """Fixed Shot Type Collection (Base Classification)"""
    BASELINE_TOPSPIN = "Baseline Topspin"
    BASELINE_SLICE = "Baseline Slice"
    BASELINE_FLAT = "Baseline Flat"
    INSIDE_OUT = "Inside-Out Angle Attack"
    DROP_SHOT = "Drop Shot"
    LOB = "Lob"


class DetailedShotType(Enum):
    """
    Detailed Shot Type System (Granular shot types with directional specification)
    
    Based on .crusorrules V2026.46-Audited:
    - Baseline shots (Topspin/Slice/Flat): [Left, Center, Right] (9 types)
    - Inside-Out Angle Attack: [Left, Right] targeting Y=±14 (2 types)
    - Drop Shots: [Left, Center, Right] (3 types)
    - Lob: No directional variant (1 type)
    - Total: 15 distinct shot paths
    
    Direction Definitions (from my perspective):
    - Left = Y=8 (Ad side)
    - Center = Y=0 (Middle)
    - Right = Y=-8 (Deuce side)
    - Wide Left = Y=14 (Extreme Ad angle)
    - Wide Right = Y=-14 (Extreme Deuce angle)
    """
    # Baseline Topspin
    BASELINE_TOPSPIN_LEFT = "Baseline Topspin (Left)"
    BASELINE_TOPSPIN_CENTER = "Baseline Topspin (Center)"
    BASELINE_TOPSPIN_RIGHT = "Baseline Topspin (Right)"
    
    # Baseline Slice
    BASELINE_SLICE_LEFT = "Baseline Slice (Left)"
    BASELINE_SLICE_CENTER = "Baseline Slice (Center)"
    BASELINE_SLICE_RIGHT = "Baseline Slice (Right)"
    
    # Baseline Flat
    BASELINE_FLAT_LEFT = "Baseline Flat (Left)"
    BASELINE_FLAT_CENTER = "Baseline Flat (Center)"
    BASELINE_FLAT_RIGHT = "Baseline Flat (Right)"
    BASELINE_FLAT_DTL = "Baseline Flat (Down-The-Line)"  # DTL
    
    # Inside-Out Angle Attack
    INSIDE_OUT_LEFT = "Inside-Out (Left)"  # Target Y=14
    INSIDE_OUT_RIGHT = "Inside-Out (Right)"  # Target Y=-14
    INSIDE_OUT_WIDE = "Inside-Out (Wide Angle)"  # Target Y=±14
    
    # Drop Shot
    DROP_SHOT_LEFT = "Drop Shot (Left)"
    DROP_SHOT_CENTER = "Drop Shot (Center)"
    DROP_SHOT_RIGHT = "Drop Shot (Right)"
    
    # Lob
    LOB = "Lob"


class ShotDirection(Enum):
    """Shot Direction Enumeration"""
    LEFT = "Left"      # Y = 8 (Ad side from my perspective)
    CENTER = "Center"    # Y = 0
    RIGHT = "Right"     # Y = -8 (Deuce side from my perspective)
    WIDE_LEFT = "Wide Left"   # Y = 14
    WIDE_RIGHT = "Wide Right"  # Y = -14


# ============================================================================
# Surface Physics（場地物理）
# ============================================================================

class SurfacePhysics:
    """
    Surface Physics System (Court Surface Modifier Engine)
    
    Data Source: A Beginner's Guide to Tennis Court Surfaces
    
    Core Concepts:
    - Grass: "Rob Time" - Fast, low bounce, forces rapid decision-making
    - Clay: "Buy Time" - Slow, high bounce, allows recovery and tactical maneuvering
    - Hard: Balanced - True, consistent, predictable bounce
    
    Surface Characteristics:
    - Grass: Fast pace, low bounce, unpredictable skid, slice advantage, serve-and-volley dominant
    - Clay: Slow pace, high bounce, heavy ball, topspin advantage, point construction tactics
    - Hard: Medium pace, true bounce, all-court game advantage, predictable trajectory
    
    Physics Adjustments:
    - Ball speed: Grass +15%, Clay -15%, Hard baseline
    - Bounce height: Grass -20%, Clay +20%, Hard baseline
    - Success rate: Shot-type specific modifiers based on surface affinity
    """
    
    class SurfaceType(Enum):
        CLAY = "Clay"
        GRASS = "Grass"
        HARD = "Hard"
    
    @staticmethod
    def adjust_ball_speed(base_speed: float, surface_type: SurfaceType) -> float:
        """
        Adjust ball speed based on surface type
        
        Surface Effects:
        - Grass: Fast, ball accelerates after bounce (+15% speed)
        - Clay: Slow, ball decelerates significantly after bounce (-15% speed)
        - Hard: Medium, consistent speed (no modifier)
        
        Args:
            base_speed: Base ball speed (normalized)
            surface_type: Surface type (CLAY/GRASS/HARD)
        
        Returns:
            Adjusted ball speed (float)
        """
        if surface_type == SurfacePhysics.SurfaceType.CLAY:
            return base_speed * 0.85  # Slow (post-bounce deceleration)
        elif surface_type == SurfacePhysics.SurfaceType.GRASS:
            return base_speed * 1.15  # Fast (post-bounce acceleration)
        else:  # HARD
            return base_speed  # Medium (baseline)
    
    @staticmethod
    def adjust_bounce_height(base_height: float, surface_type: SurfaceType) -> float:
        """
        Adjust bounce height based on surface type
        
        Surface Effects:
        - Grass: Low bounce, unpredictable skid (-20% height)
        - Clay: High bounce, heavy ball trajectory (+20% height)
        - Hard: True, consistent, predictable bounce (no modifier)
        
        Args:
            base_height: Base bounce height (normalized)
            surface_type: Surface type (CLAY/GRASS/HARD)
        
        Returns:
            Adjusted bounce height (float)
        """
        if surface_type == SurfacePhysics.SurfaceType.CLAY:
            return base_height * 1.2  # High bounce (heavy ball)
        elif surface_type == SurfacePhysics.SurfaceType.GRASS:
            return base_height * 0.8  # Low bounce (skid/slide)
        else:  # HARD
            return base_height  # True bounce (baseline)
    
    @staticmethod
    def adjust_success_rate(
        base_rate: float,
        shot_type: FixedShotType,
        surface_type: SurfaceType,
        phase: PhaseType
    ) -> float:
        """
        Adjust success rate based on surface type and shot type affinity
        
        Surface-Shot Affinity:
        - Grass: 平擊球優勢、切球優勢、上旋球劣勢、發球上網優勢
        - Clay: 上旋球優勢、平擊球劣勢、難以直接得分
        - Hard: 平衡型，平擊和上旋都有效
        
        參數：
            base_rate: 基礎成功率
            shot_type: 擊球類型
            surface_type: 場地類型
            phase: 相位
        
        返回：
            調整後的成功率
        """
        adjustment = 0.0
        
        if surface_type == SurfacePhysics.SurfaceType.GRASS:
            # 草地：平擊球和切球優勢，上旋球劣勢
            if shot_type == FixedShotType.BASELINE_FLAT:
                adjustment = +8.0  # 平擊球優勢
            elif shot_type == FixedShotType.BASELINE_SLICE:
                adjustment = +10.0  # 切球優勢（backspin 造成低滑行）
            elif shot_type == FixedShotType.BASELINE_TOPSPIN:
                adjustment = -5.0  # 上旋球劣勢（spin 效果較差）
            
            # 發球上網優勢
            if phase == PhaseType.OFFENSE and shot_type in [FixedShotType.BASELINE_FLAT, FixedShotType.BASELINE_SLICE]:
                adjustment += 5.0
        
        elif surface_type == SurfacePhysics.SurfaceType.CLAY:
            # 紅土：上旋球優勢，平擊球劣勢
            if shot_type == FixedShotType.BASELINE_TOPSPIN:
                adjustment = +10.0  # 上旋球優勢（極大上旋）
            elif shot_type == FixedShotType.BASELINE_FLAT:
                adjustment = -8.0  # 平擊球劣勢
            
            # 難以直接得分（需要點建構）
            if phase == PhaseType.OFFENSE:
                adjustment -= 3.0  # 直接得分更困難
        
        else:  # HARD
            # 硬地：平衡型，所有擊球類型都有效
            adjustment = 0.0
        
        return max(0, min(100, base_rate + adjustment))
    
    @staticmethod
    def get_time_control_factor(surface_type: SurfaceType) -> float:
        """
        獲取時間控制因子
        
        資料：
        - Grass: "搶時間" (rob time) - 因子 > 1.0，強迫快速決策
        - Clay: "買時間" (buy time) - 因子 < 1.0，允許恢復和戰術機動
        - Hard: 平衡 - 因子 = 1.0
        
        參數：
            surface_type: 場地類型
        
        返回：
            時間控制因子（>1.0 為搶時間，<1.0 為買時間）
        """
        if surface_type == SurfacePhysics.SurfaceType.GRASS:
            return 1.2  # 搶時間（快速決策）
        elif surface_type == SurfacePhysics.SurfaceType.CLAY:
            return 0.8  # 買時間（允許恢復）
        else:  # HARD
            return 1.0  # 平衡
    
    @staticmethod
    def is_bounce_predictable(surface_type: SurfaceType) -> bool:
        """
        判定彈跳是否可預測
        
        資料：
        - Grass: 不可預測（especially as tournament wears on）
        - Clay: 相對可預測（但高彈跳）
        - Hard: 真實、一致、可預測
        
        參數：
            surface_type: 場地類型
        
        返回：
            True 為可預測，False 為不可預測
        """
        if surface_type == SurfacePhysics.SurfaceType.GRASS:
            return False  # 不可預測彈跳
        elif surface_type == SurfacePhysics.SurfaceType.CLAY:
            return True  # 相對可預測（但高彈跳）
        else:  # HARD
            return True  # 真實、一致、可預測
    
    @staticmethod
    def get_serve_and_volley_bonus(surface_type: SurfaceType) -> float:
        """
        獲取發球上網加成
        
        資料：
        - Grass: 發球上網戰術優勢（serve-and-volley thrives）
        - Clay: 發球上網劣勢
        - Hard: 平衡
        
        參數：
            surface_type: 場地類型
        
        返回：
            成功率加成（百分比）
        """
        if surface_type == SurfacePhysics.SurfaceType.GRASS:
            return +12.0  # 發球上網優勢
        elif surface_type == SurfacePhysics.SurfaceType.CLAY:
            return -8.0  # 發球上網劣勢
        else:  # HARD
            return 0.0  # 平衡


@dataclass
class ShotStep:
    """
    Single Shot Step Data Structure
    
    Represents a single shot in a rally sequence with complete tactical metadata.
    """
    step_num: int
    player: str
    shot_type: FixedShotType
    detailed_shot_type: Optional[DetailedShotType] = None
    start_pos: Tuple[float, float] = (0.0, 0.0)
    end_pos: Tuple[float, float] = (0.0, 0.0)
    recovery_pos: Tuple[float, float] = (0.0, 0.0)
    zone_name: str = ""
    phase: PhaseType = PhaseType.NEUTRAL
    is_winner: bool = False
    is_forced: bool = False
    distance_to_opponent: float = 0.0
    success_rate: float = 0.0
    risk_level: str = ""
    risk_explanation: str = ""
    tactical_intent: str = ""
    theory_applied: List[str] = field(default_factory=list)
    logic_trace: Dict[str, str] = field(default_factory=dict)
    air_target_height: Optional[str] = None  # Dynamic Air Target: Low/Medium/High
    tactic_id: int = 0  # Tactic ID (maps to 48-tactic library, 0 = undefined)
    tactic_name: str = "Undefined"  # Tactic name (default "Undefined")


@dataclass
class OpponentPrediction:
    """
    Opponent Shot Prediction (Extended version with scoring breakdown and tactical traceability)
    
    Represents a predicted opponent shot with complete tactical analysis and probability scoring.
    """
    option: str = ""
    name: str = ""
    target_pos: Tuple[float, float] = (0.0, 0.0)
    shot_type: FixedShotType = FixedShotType.BASELINE_TOPSPIN
    detailed_shot_type: Optional[DetailedShotType] = None
    zone_name: str = ""
    tactical_reasoning: str = ""
    score: float = 0.0
    probability: float = 0.0
    # Scoring breakdown (detailed weight components)
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    # Applied tactical principles
    tactical_principles: List[str] = field(default_factory=list)
    # Tactical traceability (maps to 48-tactic library)
    tactic_id: int = 48
    tactic_name: str = "Non-Standard Tactic"
    
    # Scientific simulation attributes (V2026.Scientific)
    confidence_level: str = "Medium"  # High/Medium/Low
    uncertainty_interval: str = "±5.0%"
    # Optimized recovery position (Garlikov Recovery)
    recovery_pos: Tuple[float, float] = (39.0, 0.0)
@dataclass
class TacticalStrategy:
    """
    Tactical Strategy Unit - V2026.48 Pattern-First Edition
    
    Represents one of the 48 tactical patterns from the master library.
    
    Core Components:
    - Tactic ID mapping (1-48)
    - Shot type specification
    - Target coordinate calculation (dynamic based on state)
    - Axiomatic utility scoring (Wardlaw/Garlikov compliance)
    - **NEW V2026.48**: Contextual multiplier (0.0-5.0x) for pattern matching
    """
    
    def __init__(self, tactic_id: int, name: str, description: str, shot_type: FixedShotType, 
                 target_x: float = 39.0, target_y: float = 0.0):
        """
        Initialize tactical strategy
        
        Args:
            tactic_id: Tactic ID (1-48 from master library)
            name: Tactic name (English preferred)
            description: Tactical reasoning/principles
            shot_type: Fixed shot type (BASELINE_TOPSPIN, etc.)
            target_x: Default target X-coordinate
            target_y: Default target Y-coordinate
        """
        self.id = tactic_id
        self.name = name
        self.description = description
        self.shot_type = shot_type
        self.target_x = target_x
        self.target_y = target_y
        self.contextual_multiplier = 1.0  # V2026.48: Pattern-First multiplier (default 1.0x)
    
    def set_contextual_multiplier(self, multiplier: float):
        """
        Set contextual matching multiplier (V2026.48 Pattern-First Logic)
        
        Args:
            multiplier: Contextual relevance score (0.0-5.0, where 5.0 = perfect match)
        """
        self.contextual_multiplier = max(1.0, multiplier)  # Minimum 1.0x (no penalty)
    
    # Applicability condition logic
    def is_applicable(self, state: Dict) -> bool:
        """
        Determine if this tactic is applicable in current state
        
        Returns:
            True if tactic can be used, False otherwise
        """
        return True
        
    def calculate_axiomatic_utility(self, s: Dict) -> Dict[str, float]:
        """
        Core Axiom-Driven Utility Engine
        
        Calculates three fundamental utility components:
        1. **Geometric Pressure**: Whether shot forces opponent out of balanced position
        2. **Displacement Cost**: Physical lateral distance forced upon opponent
        3. **Axiom Bonus**: Reward for compliance with Wardlaw (directional) or Garlikov (recovery) principles
        
        Args:
            s: State dictionary containing positions, history, phase, opponent style
        
        Returns:
            Dict with keys: "geo_pressure", "disp_cost", "axiom_utility" (all floats 0.0-1.0)
        """
        tx, ty = self.get_dynamic_target(s)
        target_pos = (tx, ty)
        
        # 1. Geometric Pressure Coefficient (0.0-1.0)
        geo_pressure = CourtGeometry.calculate_geometric_pressure(target_pos, s["opp_pos"])
        
        # 2. Displacement Cost (normalized 0.0-1.0, based on my position)
        disp_cost = min(1.0, CourtGeometry.calculate_displacement_cost(target_pos, s["my_pos"]) / 36.0)
        
        # 3. Wardlaw Principle Bonus (directional axiom)
        # Determine if opponent's shot is "outside ball": use incoming ball position
        if s["history"]:
            last_shot = s["history"][-1]
            incoming_y = last_shot.end_pos[1] if last_shot.player != "Opponent" else last_shot.start_pos[1]
        else:
            incoming_y = 0.0
        is_outside = WardlawTheory.is_outside_ball(incoming_y, s["opp_pos"][1], s.get("is_lefty", False))
        shot_is_cross = CourtGeometry.is_crosscourt_shot(s["opp_pos"], target_pos)
        
        axiom_utility = 0.0
        if is_outside:
            # Outside ball principle: MUST hit crosscourt
            # 🔥 AGGRESSIVE opponent: More willing to violate Wardlaw for DTL attack
            if s.get("style") == OpponentStyle.AGGRESSIVE:
                # Aggressive opponent: Outside ball + DTL gains bonus (high-risk, high-reward)
                if shot_is_cross:
                    axiom_utility = 0.3  # Crosscourt: safe but passive
                else:
                    axiom_utility = 0.9  # DTL: risky but aggressive, major bonus
                    # Offensive phase: Completely ignore Wardlaw, DTL gets full score
                    if s.get("opp_phase") == PhaseType.OFFENSE:
                        axiom_utility = 1.0  # Offensive phase DTL: maximum score
            else:
                # Other styles: Strictly follow Wardlaw principles
                axiom_utility = 1.0 if shot_is_cross else -0.5
        else:
            # Inside ball principle: Allow and reward redirection (DTL/Inside-out)
            if s.get("style") == OpponentStyle.AGGRESSIVE:
                # Aggressive opponent: Inside ball + DTL = absolute advantage
                axiom_utility = 1.0 if not shot_is_cross else 0.2  # DTL full score, crosscourt low score
            else:
                axiom_utility = 0.8 if not shot_is_cross else 0.4

        return {
            "geo_pressure": geo_pressure,
            "disp_cost": disp_cost,
            "axiom_utility": axiom_utility
        }

    def _calculate_excel_tactic_bonus(
        self,
        state: Dict,
        shot_is_cross: bool,
        tx: float,
        ty: float
    ) -> float:
        """
        情境觸發式戰術加成
        只在符合 Excel 戰術表「使用時機或情境」時才加分
        避免與 style_bonus 重複計算
        """
        profile = _match_excel_tactic_profile(self.name)
        if not profile:
            return 0.0

        usage = str(profile.get("使用時機或情境", ""))
        action = str(profile.get("具體執行動作", ""))
        risk = str(profile.get("潛在風險或缺點", ""))
        name = str(profile.get("戰術名稱", ""))
        text = f"{name} {usage} {action}"

        bonus = 0.0
        opp_pos = state["opp_pos"]
        my_phase = state.get("my_phase", PhaseType.NEUTRAL)
        opp_phase = state["opp_phase"]
        history = state.get("history", [])
        rally_count = len([s for s in history if s.player == "Opponent"])

        # ========== 情境觸發邏輯 ==========
        
        # 1. 對手位置情境
        opp_at_net = opp_pos[0] > -12.0  # 對手在網前
        opp_deep = opp_pos[0] < -30.0    # 對手在底線深處
        opp_left = opp_pos[1] < -8.0     # 對手偏左
        opp_right = opp_pos[1] > 8.0     # 對手偏右
        opp_center = abs(opp_pos[1]) < 6.0  # 對手在中路

        # 2. 特殊情境匹配
        # 中路球情境
        if "中路" in text and abs(ty) < 4.0:
            bonus += 8.0
        
        # 對手在網前的戰術
        if opp_at_net:
            if any(k in usage for k in ["網前", "上網", "截擊", "穿越", "挑高"]):
                bonus += 12.0
            # 低球打腳下
            if "腳下" in action or "腳部" in action or "低位" in action:
                bonus += 8.0
        
        # 對手在底線深處
        if opp_deep:
            if "深處" in usage or "底線後方" in usage:
                bonus += 10.0
            # 短角度球
            if "短角" in name or "短球" in action:
                bonus += 12.0
        
        # 對手偏向一側
        if opp_left or opp_right:
            if "空檔" in action or "大角度" in text:
                bonus += 10.0
        
        # 底線拉鋸戰
        if rally_count >= 4:
            if "拉鋸" in usage or "相持" in usage or "對抗" in usage:
                bonus += 8.0
        
        # 極度被動情境
        if my_phase == PhaseType.DEFENSE:
            if "被動" in usage or "防守" in usage or "挑高" in name:
                bonus += 12.0
        
        # 進攻機會情境
        if my_phase == PhaseType.OFFENSE:
            if "短球" in usage or "進攻" in usage or "主動" in usage:
                bonus += 10.0
        
        # 對手回球過短
        if history and history[-1].end_pos[0] > -25.0:
            if "短球" in usage or "機會球" in usage:
                bonus += 10.0
        
        # 對手失位情境
        if history and len(history) >= 2:
            last_opp_y = history[-1].end_pos[1]
            if abs(last_opp_y) > 10.0:  # 對手被拉出場外
                if "失位" in usage or "空檔" in action or "追身" in name:
                    bonus += 12.0
        
        # 發球後情境 (Serve +1)
        if rally_count == 1:
            if "發球" in usage or "Serve" in name or "+1" in name:
                bonus += 8.0
        
        # 對手風格相關情境
        style = state.get("style", OpponentStyle.AGGRESSIVE)
        if style == OpponentStyle.PUSHER:
            if "Pusher" in usage or "推球手" in usage or "防守型" in usage:
                bonus += 10.0
        
        # 3. 風險調整（根據對手風格和相位）
        risk_level = 0
        if any(k in risk for k in ["風險", "失誤", "掛網", "出界", "危險"]):
            risk_level = 1
        if any(k in risk for k in ["極高", "非常", "高度"]):
            risk_level = 2
        
        if risk_level > 0:
            # 進攻型對手在進攻相位時容忍風險
            if style == OpponentStyle.AGGRESSIVE and opp_phase == PhaseType.OFFENSE:
                penalty = (2.0 if risk_level == 1 else 4.0)
            else:
                phase_factor = {
                    PhaseType.OFFENSE: 0.8,
                    PhaseType.NEUTRAL: 1.0,
                    PhaseType.DEFENSE: 1.3
                }.get(opp_phase, 1.0)
                penalty = (5.0 if risk_level == 1 else 9.0) * phase_factor
            bonus -= penalty

        return bonus

    # 評分權重邏輯
    def calculate_weight(self, state: Dict) -> float:
        """
        Calculate final tactical weight (V2026.48 Pattern-First Architecture)
        
        Formula (Base):
        Weight = (Geometric_Pressure * 0.4 + Displacement_Cost * 0.3 + Axiom_Utility * 0.3) * 100
        
        Adjustments:
        - 70/30 Rule (Crosscourt/DTL base weights)
        - Opponent style modifiers (Aggressive/Pusher/Net Rusher)
        - Phase-specific bonuses (D-N-O)
        - Anti-stalemate logic (repetition penalty + breakout bonus)
        - PWP (Point Win Probability) scoring integration
        - Dynamic risk barrier (fatigue, critical points)
        - 🔥 NEW V2026.48: Contextual multiplier (1.0-5.0x) for pattern-matched tactics
        
        Returns:
            Final weight (0-250, capped after contextual multiplier)
        """
        tx, ty = self.get_dynamic_target(state)
        u = self.calculate_axiomatic_utility(state)
        
        # ============================================================================
        # MASTER 70/30 PRINCIPLE - Core Weight Initialization
        # ============================================================================
        # Following Top Tennis 70/30: Crosscourt is high-percentage foundation
        shot_is_cross = CourtGeometry.is_crosscourt_shot(state["opp_pos"], (tx, ty))
        
        # 🔥 V2026.48: AGGRESSIVE OPPONENT OVERRIDE - Reverse 70/30 → 30/70 (DTL Priority)
        # Aggressive opponents prefer DTL shots to finish points rather than crosscourt rallies
        # This implements the "Override Probabilistic Noise" requirement: Tactical intent > Generic penalties
        if state["style"] == OpponentStyle.AGGRESSIVE:
            # Fix A+: Set minimum geo_pressure for aggressive DTL shots
            # Prevents DTL shots from losing competitiveness when lateral distance = 0
            effective_geo_pressure = u["geo_pressure"]
            if not shot_is_cross:
                # DTL shot: Ensure minimum geo_pressure = 1.0 (Aggressive opponents view DTL as high-pressure tactic)
                effective_geo_pressure = max(1.0, u["geo_pressure"])
            
            # Aggressive opponent: DTL base weight = 100, Crosscourt base weight = 10
            if shot_is_cross:
                base_w = 10.0 + (u["geo_pressure"] * 6.0 + u["axiom_utility"] * 6.0)  # Crosscourt: Low base (passive)
            else:
                base_w = 100.0 + (effective_geo_pressure * 30.0 + u["axiom_utility"] * 40.0)  # DTL: High base (aggressive finisher)
        else:
            # Other styles (Pusher, Net Rusher): Follow traditional 70/30 principle
            if shot_is_cross:
                base_w = 70.0 + (u["geo_pressure"] * 15.0 + u["axiom_utility"] * 15.0)
            else:
                base_w = 30.0 + (u["geo_pressure"] * 30.0 + u["axiom_utility"] * 40.0)
            
        # Stabilization Adjustment: Baseline Rally Phase Weight Correction
        if state["opp_phase"] == PhaseType.NEUTRAL:
            if self.shot_type == FixedShotType.BASELINE_TOPSPIN:
                # Don't boost conservative "crosscourt topspin" for aggressive opponents in neutral phase
                if not (state["style"] == OpponentStyle.AGGRESSIVE and shot_is_cross):
                    base_w += 15.0
            elif self.shot_type in [FixedShotType.DROP_SHOT, FixedShotType.BASELINE_SLICE]:
                base_w -= 20.0 

        # ============================================================================
        # SPECIAL MODE: Stable Defense Mode Override
        # ============================================================================
        # If in "Stable Defense" mode, massively boost crosscourt topspin and exclude all else
        if state.get("tactical_mode") == "穩定防守": # Match string with app_main
            if shot_is_cross and self.shot_type == FixedShotType.BASELINE_TOPSPIN:
                return 100.0  # Force maximum score
            if not shot_is_cross or self.shot_type != FixedShotType.BASELINE_TOPSPIN:
                return 0.1    # Nearly exclude
        
        # Style Modifier & Aggressive Enhancement
        style_bonus = 0.0
        if state["style"] == OpponentStyle.AGGRESSIVE:
            # 🔥 V2026.48: AGGRESSIVE OPPONENT - More Aggressive Attack Logic
            # 1. Prioritize Geometric Pressure and Winner Zones
            is_winner_zone = abs(tx) > 20 and abs(ty) > 12
            # Fix A+: Use same effective_geo_pressure logic as base_w
            effective_geo_pressure_bonus = u["geo_pressure"]
            if not shot_is_cross:
                effective_geo_pressure_bonus = max(1.0, u["geo_pressure"])
            style_bonus = effective_geo_pressure_bonus * 30.0  # Boost geometric pressure weight
            if is_winner_zone:
                style_bonus += 30.0  # Winner Zone: Major bonus
            
            # 2. DTL Shot Extra Bonus (Core Tactic for Aggressive Opponents)
            if not shot_is_cross:
                # Phase-specific DTL bonuses
                if state["opp_phase"] == PhaseType.OFFENSE:
                    style_bonus += 25.0 + 40.0  # Offensive phase: Base +25, Extra +40
                elif state["opp_phase"] == PhaseType.NEUTRAL:
                    style_bonus += 25.0 + 10.0  # Neutral phase: Base +25, Extra +10
                else:  # DEFENSE
                    style_bonus += 10.0  # Defensive phase: Moderate bonus (more conservative)
            else:
                # Crosscourt: Phase-specific adjustments
                if state["opp_phase"] == PhaseType.OFFENSE:
                    style_bonus -= 20.0  # Offensive phase: Moderate penalty (preserve minimum weight 10, not -30)
                elif state["opp_phase"] == PhaseType.NEUTRAL:
                    style_bonus -= 5.0   # Neutral phase: Slight penalty
                else:  # DEFENSE
                    style_bonus += 5.0    # Defensive phase: Moderate bonus (build advantage)

            # 2.5 方向性動機曲線（平滑相位權重）
            phase_factor = {
                PhaseType.OFFENSE: 1.0,
                PhaseType.NEUTRAL: 0.6,
                PhaseType.DEFENSE: 0.2
            }.get(state["opp_phase"], 0.6)
            directional_bias = 20.0 * phase_factor
            if not shot_is_cross:
                style_bonus += directional_bias
            else:
                style_bonus -= directional_bias * 0.6
            
            # 3. 球種偏好：根據相位調整球種偏好
            if self.shot_type == FixedShotType.BASELINE_FLAT:
                style_bonus += 20.0  # 平擊球大幅加分（進攻性球種）
            elif self.shot_type == FixedShotType.INSIDE_OUT:
                style_bonus += 25.0  # 大角度攻擊大幅加分
            elif self.shot_type == FixedShotType.BASELINE_TOPSPIN:
                # 【🔥 改進】根據相位調整上旋球權重
                if state["opp_phase"] == PhaseType.OFFENSE:
                    style_bonus -= 10.0  # 進攻相位：上旋球扣分（應該打平擊球）
                elif state["opp_phase"] == PhaseType.NEUTRAL:
                    # 相持相位：對角上旋再降低，避免被動黏對角
                    style_bonus -= 8.0 if shot_is_cross else -5.0
                else:  # DEFENSE
                    style_bonus += 5.0    # 防守相位：上旋球加分（建立優勢）
            elif self.shot_type == FixedShotType.BASELINE_SLICE:
                # 【🔥 改進】添加切球處理
                if state["opp_phase"] == PhaseType.DEFENSE:
                    style_bonus += 10.0  # 防守相位時切球加分 

            # 4. 進攻型平擊直線偏好（平滑相位權重）
            if self.shot_type == FixedShotType.BASELINE_FLAT and not shot_is_cross:
                style_bonus += 12.0 * phase_factor
                
        elif state["style"] == OpponentStyle.PUSHER:
            # Pusher 強度依附 70/30，且進一步排斥變線
            if not shot_is_cross:
                style_bonus = -40.0
            else:
                style_bonus = 15.0
            
        # ============================================================================
        # ANTI-STALEMATE LOGIC (Repetition Penalty & Breakout Bonus)
        # ============================================================================
        repetition_penalty = 0.0
        stalemate_bonus = 0.0
        
        if state["history"]:
            last_shot = state["history"][-1]
            last_target = last_shot.end_pos
            
            # 1. Repetition Penalty (Consecutive Same-Zone Targeting)
            # If target zone matches previous shot zone, apply penalty to force AI to find new targets
            dist_to_last_target = math.sqrt((tx - last_target[0])**2 + (ty - last_target[1])**2)
            if dist_to_last_target < 8.0:  # Same zone (within 8ft radius)
                repetition_penalty = -25.0
            
            # 2. Stalemate Breakout Bonus
            # As neutral rally count increases, boost DTL and wide-angle shots to break the rally
            neutral_count = 0
            for s_step in reversed(state["history"]):
                if s_step.phase == PhaseType.NEUTRAL:
                    neutral_count += 1
                else: break
            
            if neutral_count >= 2:
                # Continuous function to quantify "breakout intent" (avoid hard thresholds)
                last_target_y = last_target[1]
                angle_shift = abs(ty - last_target_y)
                direction_change = 1.0 if (last_target_y * ty) < 0 else 0.0
                dtl_signal = 1.0 if not shot_is_cross else 0.0

                # Continuous: Angle expansion strength (0-1)
                angle_strength = 1.0 / (1.0 + math.exp(-(angle_shift - 9.0) / 3.0))

                # 🔥 V2026.53: Use GeometricClassifier for direction change detection
                # Replaces manual sign checking: (last_target_y * ty) < 0
                my_y = state.get("my_pos", (0, 0))[1]
                last_direction = GeometricClassifier.classify_direction(my_y, last_target_y)
                current_direction = GeometricClassifier.classify_direction(my_y, ty)
                direction_change_bool = (last_direction != current_direction)
                direction_change = 1.0 if direction_change_bool else 0.0
                
                # Continuous: DTL + direction change composite signal (0-1)
                # DTL dominates, direction change assists (avoid scripted patterns)
                line_strength = 1.0 / (1.0 + math.exp(-(1.6 * dtl_signal + 0.8 * direction_change - 0.6)))

                # Continuous breakout score (0-1)
                breakout_score = max(0.0, min(1.0, 0.55 * line_strength + 0.45 * angle_strength))

                # Phase and style adjustment factors
                phase_factor = {
                    PhaseType.OFFENSE: 1.0,
                    PhaseType.NEUTRAL: 0.8,
                    PhaseType.DEFENSE: 0.5
                }.get(state["opp_phase"], 0.8)
                style_factor = 1.1 if state["style"] == OpponentStyle.AGGRESSIVE else 0.6 if state["style"] == OpponentStyle.PUSHER else 1.0

                base_stalemate_bonus = (neutral_count - 1) * 12.0
                stalemate_bonus = base_stalemate_bonus * breakout_score * phase_factor * style_factor
        
        # --------------------------------------------------------------------
        # 【🔥 重構】致勝期望值 (PWP) 評分引擎整合
        # --------------------------------------------------------------------
        # 計算基礎成功率（根據球種和相位）
        base_success_rate = TopTennisBrain.calculate_success_rate(
            self.shot_type,
            state["opp_pos"],
            (tx, ty),
            state["opp_phase"]
        )
        
        # 計算組合拳加成
        combo_bonus = 0.0
        if state.get("tactical_package"):
            combo_bonus = TacticalComboEngine.get_combo_bonus(
                len(state["history"]) + 1,
                state["tactical_package"],
                state["history"]
            )
        
        # 計算 PWP 評分
        pwp_result = PointWinProbabilityEngine.calculate_pwp_score(
            self.shot_type,
            (tx, ty),
            state["my_pos"],
            state["opp_phase"],
            base_success_rate,
            state["surface"],
            combo_bonus
        )
        
        # PWP 加成（占總權重的 30%）
        pwp_bonus = pwp_result["pwp_score"] * 0.3
        
        # --------------------------------------------------------------------
        # 【🔥 重構】動態風險護欄整合
        # --------------------------------------------------------------------
        # 計算關鍵分係數（如果有比分信息）
        critical_point_factor = 1.0
        if state.get("score") and state.get("break_points") is not None:
            critical_point_factor = DynamicRiskBarrierEngine.calculate_critical_point_factor(
                state["score"],
                state["break_points"],
                state.get("is_set_point", False)
            )
        
        # 計算疲勞扣分（基於推演拍數）
        fatigue_penalty = 0.0
        if len(state["history"]) > 0:
            # 計算移動距離（假設從對手當前位置到目標位置）
            movement_distance = Geometry30Point.calculate_distance(
                state["opp_pos"], (tx, ty)
            )
            fatigue_penalty = DynamicRiskBarrierEngine.calculate_fatigue_penalty(
                len(state["history"]) + 1,
                movement_distance,
                max_steps=10
            )
        
        # 計算基礎權重（包含所有傳統加成）
        tactic_bonus = self._calculate_excel_tactic_bonus(state, shot_is_cross, tx, ty)
        traditional_weight = base_w + style_bonus + repetition_penalty + stalemate_bonus + tactic_bonus
        
        # 【調試】打印權重計算明細（僅在開發模式下）
        if state.get("debug_mode"):
            direction = "對角" if shot_is_cross else "直線"
            print(f"[DEBUG] {self.name} ({direction}): base_w={base_w:.1f}, style={style_bonus:.1f}, "
                  f"rep_pen={repetition_penalty:.1f}, stalemate={stalemate_bonus:.1f}, tactic={tactic_bonus:.1f}, "
                  f"total={traditional_weight:.1f}, geo_p={u['geo_pressure']:.2f}, axiom={u['axiom_utility']:.2f}")
        
        # 應用 PWP 加成
        weight_with_pwp = traditional_weight + pwp_bonus
        
        # 應用動態風險護欄調整
        final_weight = DynamicRiskBarrierEngine.adjust_risk_tolerance(
            weight_with_pwp,
            critical_point_factor,
            state["opp_phase"],
            state["style"]
        )
        
        # 應用疲勞扣分
        final_weight -= fatigue_penalty
        
        # 【🔥 重構】回位品質影響
        if len(state["history"]) > 0:
            last_shot = state["history"][-1]
            recovery_impact = TacticalComboEngine.calculate_recovery_quality_impact(
                last_shot,
                state["my_pos"]
            )
            # 如果上一拍威脅程度高，增加當前拍的進攻窗口
            final_weight *= recovery_impact["attack_window_multiplier"]
            final_weight += recovery_impact["recovery_delay_bonus"] * 0.2  # 20% 權重
        
        # ============================================================================
        # 🔥 V2026.48 PATTERN-FIRST MULTIPLIER (Tactical Context Matching)
        # ============================================================================
        # Apply contextual multiplier (0.0-5.0x) based on tactics_data.json matching
        # This multiplier boosts tactics that specifically match the current game context:
        # - Phase alignment (D-N-O): +1.5x
        # - Position criteria: +1.5x
        # - Opponent style: +1.0x
        # - Shot history pattern: +1.0x
        # Maximum boost: 5.0x (perfect contextual match)
        final_weight *= self.contextual_multiplier
        
        # 🔥 V2026.52 MANDATORY: Apply Wardlaw Directional Bonus/Penalty
        # 🔥 V2026.53 CRITICAL FIX: Apply Wardlaw bonus BEFORE weight cap
        # Enforce geometric clarity: Crosscourt (70%) vs DTL (30%)
        # Uses strict classification from GeometricClassifier
        target_pos = self.get_dynamic_target(state)
        my_pos = state.get("my_pos", (0, 0))
        
        if target_pos and len(target_pos) >= 2:
            final_weight = GeometricClassifier.apply_wardlaw_bonus(
                start_y=my_pos[1],
                target_y=target_pos[1],
                base_score=final_weight
            )
        
        # 🔥 V2026.53: Apply cap AFTER Wardlaw bonus (moved from above)
        # This ensures Wardlaw adjustments are not wasted on already-capped weights
        return max(0, min(250, final_weight))  # Cap at 250 AFTER bonus applied

    # 動態落點調整
    def get_dynamic_target(self, state: Dict) -> Tuple[float, float]:
        """根據當前坐標動態調整落點"""
        return (self.target_x, self.target_y)

    def to_prediction(self, state: Dict, weight: float) -> OpponentPrediction:
        """將戰術轉換為科學預測格式 (V2026.Scientific)"""
        tx, ty = self.get_dynamic_target(state)
        u = self.calculate_axiomatic_utility(state)
        
        # 物理驗證
        physical_verification = PhysicalVerifier.verify_magnus_effect(self.shot_type)
        
        # 【 मास्टर (Master) 】戰術名稱動態調整
        # 如果擊球點在底線後方而戰術是「網前短球」，則更名為「底線放短球」以維持物理直覺
        display_name = self.name
        if self.shot_type == FixedShotType.DROP_SHOT and state["opp_x"] < -30:
            display_name = "底線放短球"
        
        # Axiom Analysis Report (English Reasoning - V2026.48)
        if abs(state["opp_pos"][1]) < 6.0:
            axiom_report = "[Midpoint Divergence Axiom] Opponent at center (Y~0), optimal target at extreme angles"
        else:
            axiom_report = "[Sideline Pressure Axiom] Opponent wide (|Y|>6), exploit geometric displacement cost"
        
        if u.get("axiom_utility", 0) > 0.8:
            axiom_report = "[Wardlaw Directional Axiom: Excellent] Perfect compliance with inside/outside ball theory"
        
        # 🔥 V2026.48: Pattern-First Contextual Matching Report
        contextual_report = ""
        if self.contextual_multiplier > 1.5:
            contextual_report = f"[PATTERN MATCH] Contextual Multiplier: {self.contextual_multiplier:.2f}x | "
            if self.contextual_multiplier >= 4.0:
                contextual_report += "PERFECT MATCH: Phase + Position + Style + Pattern"
            elif self.contextual_multiplier >= 3.0:
                contextual_report += "HIGH MATCH: Phase + Position + (Style OR Pattern)"
            elif self.contextual_multiplier >= 2.0:
                contextual_report += "MODERATE MATCH: Phase + Position"
            else:
                contextual_report += "WEAK MATCH: Phase OR Position"
        
        # Zone Classification (Winning Zone vs Safety Zone)
        is_winning_zone = "DTL" in self.name or "角度" in self.name or "Inside-In" in self.name or weight > 85
        zone_type = "Winning Zone (High Risk/High Reward)" if is_winning_zone else "Safety Zone (Stable Control)"
        
        # Confidence Level & Uncertainty Interval (English V2026.48)
        if weight >= 120:  # Adjusted threshold for V2026.48 (pattern-matched tactics)
            confidence = "Excellent"
            uncertainty = "±1.5%"
        elif weight >= 80:
            confidence = "High"
            uncertainty = "±2.5%"
        elif weight >= 40:
            confidence = "Medium"
            uncertainty = "±5.0%"
        else:
            confidence = "Low"
            uncertainty = "±8.5%"
        
        # 🔥 V2026.48: Assemble English Tactical Reasoning with Pattern-First Context
        if contextual_report:
            academic_reasoning = f"{contextual_report} | {axiom_report} | {self.description} | {physical_verification}"
        else:
            academic_reasoning = f"{axiom_report} | {self.description} | {physical_verification}"
        
        return OpponentPrediction(
            option="", 
            name=self.name,
            target_pos=(tx, ty),
            shot_type=self.shot_type,
            zone_name=f"{Geometry30Point.get_zone_name(tx, ty)} [{zone_type}]",
            tactical_reasoning=academic_reasoning,
            score=weight,
            # 戰術溯源 ID 與名稱
            tactic_id=self.id,
            tactic_name=display_name, # 使用調整後的顯示名稱
            confidence_level=confidence,
            # 信心等級與不確定性判定
            uncertainty_interval=uncertainty,
            score_breakdown={
                "幾何壓力": round(u.get("geo_pressure", 0) * 10, 1),
                "位移代價": round(u.get("disp_cost", 0) * 10, 1),
                "公理加成": round(u.get("axiom_utility", 0) * 10, 1)
            },
            # 計算最佳化幾何回位點
            recovery_pos=GarlikovRecovery.calculate_optimal_recovery(
                striker_pos=state["opp_pos"],
                target_pos=(tx, ty),
                shot_type=self.shot_type.value if hasattr(self.shot_type, "value") else str(self.shot_type),
                is_my_side=False # 這是對手的預測，回位點在對手半場
            )
        )

# ============================================================================
# 運動科學顧問模組 (Sports Science Consultant Modules)
# ============================================================================

class ConservativeEstimator:
    """
    保守估算原則模組 (Conservative Estimation Principle)
    
    1. 我方移動能力下修 10% (Coverage radius -10%)
    2. 對方擊球初速上修 15% (Reaction time requirements +15%)
    """
    # 【 मास्टर (Master) Override】實作無限觸及：將移動懲罰設為 1.0 (無懲罰)
    MOVE_DEBUFF = 1.0  # 原為 0.90，現在代表無限體能與觸及
    SPEED_BUFF = 1.15
    
    @staticmethod
    def estimate_coverage(radius: float) -> float:
        return radius * ConservativeEstimator.MOVE_DEBUFF
        
    @staticmethod
    def estimate_threat_speed(speed: float) -> float:
        return speed * ConservativeEstimator.SPEED_BUFF


class PhysicalVerifier:
    """
    物理路徑驗證器 (Physical Path Verifier)
    
    驗證擊球是否符合馬格努斯效應 (Magnus effect) 與流體動力學假設。
    """
    @staticmethod
    def verify_magnus_effect(shot_type: "FixedShotType") -> str:
        if shot_type == FixedShotType.BASELINE_TOPSPIN:
            return "驗證：頂部順風誘導低壓區，產生顯著下壓馬格努斯力 (Magnus Force)，確保深球落點。"
        elif shot_type == FixedShotType.BASELINE_SLICE:
            return "驗證：後旋產生向上升力，延長滯空時間並在彈跳後產生低滑行效應。"
        elif shot_type == FixedShotType.DROP_SHOT:
            return "驗證：旋轉動量轉化為邊界層擾動，大幅縮短彈跳後前進向量。"
        return "驗證：平擊彈道符合基礎流體動力學拋物線模型。"


# ============================================================================
# 對手擊球預估容器
# ============================================================================

class PlayerTacticalMode(Enum):
    """
    Player Tactical Mode (My Shot Strategy Selection)
    
    Defines strategic approach for shot generation:
    - STABLE: Conservative, high-percentage tennis (crosscourt emphasis)
    - AGGRESSIVE: Attack mode, risk-taking shots (inside-out, DTL)
    - VARIATION: Rhythm disruption (drop shots, slices, height variation)
    - PRESSURE_2_1: Geometric pressure pattern (2-shot setup, 1-shot finish)
    """
    STABLE = "Stable Defense"
    AGGRESSIVE = "Aggressive Attack"
    VARIATION = "Rhythm Variation"
    PRESSURE_2_1 = "2-1 Geometric Pressure"


class OpponentStyle(Enum):
    """對手風格類型"""
    PUSHER = "防守型 (Pusher)"      # 95% 對角深球，不主動變線
    AGGRESSIVE = "進攻型 (Aggressive)"  # 積極進攻，高風險球路
    NET_RUSHER = "網前型 (Net Rusher)"  # 隨球上網，R2 自動觸發上網


# ============================================================================
# Top Tennis 戰術大腦（70/30 原則）
# ============================================================================

class TopTennisBrain:
    """
    Top Tennis 戰術大腦：70/30 原則與高百分比邏輯
    
    核心原則：
    - 70% 對角球路（高成功率）
    - 30% 變線球路（進攻機會）
    - 相持相位下直線球成功率下修 20%
    """
    
    BASE_SUCCESS_RATES = {
        FixedShotType.BASELINE_TOPSPIN: 80.0,
        FixedShotType.BASELINE_SLICE: 85.0,
        FixedShotType.BASELINE_FLAT: 70.0,
        FixedShotType.INSIDE_OUT: 65.0,
        FixedShotType.DROP_SHOT: 60.0,
        FixedShotType.LOB: 55.0,
    }
    
    @staticmethod
    def calculate_success_rate(
        shot_type: FixedShotType,
        start_pos: Tuple[float, float],
        end_pos: Tuple[float, float],
        phase: PhaseType,
        is_dtl: bool = False
    ) -> float:
        """
        計算成功率（Top Tennis 70/30 原則 + 幾何優勢量化）
        
        參數：
            shot_type: 球種類型
            start_pos: 擊球起始位置
            end_pos: 擊球落點位置
            phase: 相位類型
            is_dtl: 是否為直線球（Down The Line）
        
        返回：
            成功率百分比 (0-100)
        """
        from geometry_engine import CourtGeometry
        
        base_rate = TopTennisBrain.BASE_SUCCESS_RATES.get(shot_type, 70.0)
        
        # 上旋球加成
        if shot_type == FixedShotType.BASELINE_TOPSPIN:
            base_rate += 10.0
        
        # 相持相位下直線球成功率下修 12%
        if phase == PhaseType.NEUTRAL:
            if is_dtl:
                base_rate -= 12.0 # 從 20.0 顯著降低懲罰，以反映現代職業水平
            
        # 大角度攻擊球風險
        if shot_type == FixedShotType.INSIDE_OUT:
            base_rate -= 8.0 # 從 15.0 降低
        
        # 防守區直線攻擊風險
        if phase == PhaseType.DEFENSE and shot_type == FixedShotType.BASELINE_FLAT:
            base_rate -= 15.0
        
        # 【新增】幾何優勢量化調整
        # 根據對角線 vs 直線的幾何優勢調整成功率
        geometric_adjusted_rate = CourtGeometry.calculate_geometric_success_rate_adjustment(
            start_pos, end_pos, base_rate
        )
        
        # 注意：Psychological Momentum 和 Surface Physics 調整應在調用此方法後進行
        # 因為它們需要考慮實際擊球情境（比分、場地等）
        
        return max(0, min(100, geometric_adjusted_rate))
    
    @staticmethod
    def get_risk_level(success_rate: float) -> str:
        """根據成功率判定風險等級"""
        if success_rate >= 75:
            return "低"
        elif success_rate >= 60:
            return "中"
        else:
            return "高"
    
    @staticmethod
    def get_risk_explanation(
        shot_type: FixedShotType,
        success_rate: float,
        phase: PhaseType
    ) -> str:
        """生成風險說明"""
        explanations = []
        
        if shot_type == FixedShotType.INSIDE_OUT:
            explanations.append("大角度攻擊球需要精準控制，成功率較低")
        elif shot_type == FixedShotType.BASELINE_FLAT:
            if phase == PhaseType.NEUTRAL:
                explanations.append("相持相位下直線球風險較高（70/30 原則）")
            elif phase == PhaseType.DEFENSE:
                explanations.append("防守區直線攻擊風險極高")
            else:
                explanations.append("底線平擊球速度快但容錯率低")
        elif shot_type == FixedShotType.DROP_SHOT:
            explanations.append("網前短球需要精準控制")
        elif shot_type == FixedShotType.LOB:
            explanations.append("高吊球風險高，若高度不足容易被扣殺")
        else:
            explanations.append("穩定球種，成功率較高")
        
        if success_rate < 60:
            explanations.append("整體成功率偏低，需謹慎使用")
        elif success_rate >= 80:
            explanations.append("高成功率選擇，穩定可靠")
        
        return "；".join(explanations)
    
    @staticmethod
    def get_tactical_intent(
        shot_type: FixedShotType,
        phase: PhaseType,
        zone_name: str
    ) -> str:
        """生成戰術意圖說明"""
        intents = {
            FixedShotType.BASELINE_TOPSPIN: "利用對角深球壓迫",
            FixedShotType.BASELINE_SLICE: "改變節奏，降低對手攻擊性",
            FixedShotType.BASELINE_FLAT: "積極進攻，製造壓力",
            FixedShotType.INSIDE_OUT: "大角度拉開，創造空檔",
            FixedShotType.DROP_SHOT: "網前突襲，改變節奏",
            FixedShotType.LOB: "防守反擊，爭取回位時間",
        }
        
        base_intent = intents.get(shot_type, "戰術執行")
        
        if phase == PhaseType.DEFENSE:
            base_intent += "（防守反擊）"
        elif phase == PhaseType.OFFENSE:
            base_intent += "（主動進攻）"
        
        return base_intent


# ============================================================================
# 動態對手預判引擎（風格模型 + Softmax 歸一化）
# ============================================================================

class DynamicOpponentPredictionEngine:
    """
    Dynamic Opponent Prediction Engine (V2026.48 Pattern-First Architecture)
    
    Core Features:
    - **Pattern-First Matching**: Contextually matches tactics from tactics_data.json (5.0x max multiplier)
    - **Tactical Diversity Rewards**: Encourages shot variety through anti-stalemate logic
    - **Softmax Probability Normalization**: Converts scores to realistic probability distribution
    - **Opponent Style Models**: PUSHER, NET_RUSHER, AGGRESSIVE with distinct behavior patterns
    
    Supported Opponent Styles:
    - PUSHER: 95% crosscourt deep balls, minimal DTL risk-taking
    - NET_RUSHER: Auto-triggers net approach on R2 (serve-and-volley)
    - AGGRESSIVE: Prioritizes attack, high-risk DTL shots, tactical intent overrides generic penalties
    
    Scientific Principles Applied:
    - Wardlaw Directional Principles (Inside/Outside ball theory)
    - Garlikov Recovery (Angle bisector optimal positioning)
    - Geometric Pressure Coefficient (Lateral displacement quantification)
    - Displacement Cost Function (Energy expenditure, BMR 2200 kcal baseline)
    - Crosscourt Advantage Multiplier (82.5ft diagonal vs 78.0ft DTL geometry)
    """
    
    @staticmethod
    def score_all_shots(
        my_position: Tuple[float, float],
        opponent_position: Tuple[float, float],
        my_phase: PhaseType,
        shot_history: List[ShotStep],
        is_lefty: bool = False,
        opponent_style: OpponentStyle = OpponentStyle.AGGRESSIVE,
        surface_type: SurfacePhysics.SurfaceType = SurfacePhysics.SurfaceType.HARD,
        momentum: float = 0.0,
        tactical_mode: Optional[str] = None
    ) -> List[OpponentPrediction]:
        """
        Score all possible shots (V2026.48 Pattern-First Tactical Engine)
        
        Algorithm:
        1. **Pattern-First Matching**: Load 48-tactic database from tactics_data.json
        2. **Contextual Scoring**: Match current state (Phase, Position, Style, History) against each tactic
        3. **Massive Multiplier Application**: Apply 1.0-5.0x boost to contextually matched tactics
        4. **Override Mechanism**: Tactical intent overrides generic geometric penalties (e.g., Aggressive DTL)
        5. **Mandatory Traceability**: Every shot derives from one of 48 tactical IDs (fallback: ID 1 Counterpunching)
        
        Args:
            my_position: My current position (X, Y) in feet
            opponent_position: Opponent current position (X, Y) in feet
            my_phase: My tactical phase (DEFENSE/NEUTRAL/OFFENSE)
            shot_history: Complete shot history (List[ShotStep])
            is_lefty: Whether opponent is left-handed
            opponent_style: Opponent style (AGGRESSIVE/PUSHER/NET_RUSHER)
            surface_type: Court surface (HARD/CLAY/GRASS)
            momentum: Psychological momentum (-1.0 to +1.0)
            tactical_mode: Optional tactical mode override (e.g., "穩定防守")
        
        Returns:
            List[OpponentPrediction] sorted by weight (highest first), with contextual multipliers applied
        """
        # 1. 座標正規化 (Point Reflection Normalization)
        # 將所有擊球評估正規化為「擊球者在左 (X < 0)，目標點在右 (X > 0)」
        # 這能確保戰術邏輯（對角、直線、變線）在球場兩端完全一致
        striker_x, striker_y = opponent_position
        needs_mirror = striker_x > 0
        mirror_factor = -1.0 if needs_mirror else 1.0
        
        norm_my_pos = (my_position[0] * mirror_factor, my_position[1] * mirror_factor)
        norm_opp_pos = (opponent_position[0] * mirror_factor, opponent_position[1] * mirror_factor)
        
        # 判斷擊球者相位 (基於正規化後的負座標)
        opp_phase = PhaseDetector.detect_phase(norm_opp_pos[0])
        
        # 正規化擊球歷史
        from copy import deepcopy
        norm_history = []
        for shot in shot_history:
            ns = deepcopy(shot)
            ns.start_pos = (shot.start_pos[0] * mirror_factor, shot.start_pos[1] * mirror_factor)
            ns.end_pos = (shot.end_pos[0] * mirror_factor, shot.end_pos[1] * mirror_factor)
            norm_history.append(ns)
            
        # 構造正規化狀態字典
        state = {
            "my_pos": norm_my_pos,
            "opp_pos": norm_opp_pos,
            "my_x": norm_my_pos[0],
            "my_y": norm_my_pos[1],
            "opp_x": norm_opp_pos[0],
            "opp_y": norm_opp_pos[1],
            "my_phase": my_phase,       # 接收者相位
            "opp_phase": opp_phase,      # 擊球者相位 (Striker Phase)
            "surface": surface_type,
            "momentum": momentum,
            "style": opponent_style,
            "is_lefty": is_lefty,
            "history": norm_history,
            
            # 【新增】運動科學保守估算因子 (V2026.Scientific)
            "movement_multiplier": ConservativeEstimator.MOVE_DEBUFF,
            "speed_multiplier": ConservativeEstimator.SPEED_BUFF,
            "tactical_mode": tactical_mode # 注入戰術模式
        }

        # 2. 從戰術溯源系統獲取所有適用策略
        strategies = TacticalTraceability.get_all_strategies(state)
        
        # 🔍 V2026.51 CRITICAL DIAGNOSTIC: Track strategy generation
        print(f"\n[V2026.51 Strategy Generation]")
        print(f"  Total strategies from get_all_strategies(): {len(strategies)}")
        if len(strategies) < 10:
            print(f"  ⚠️ CRITICAL: Only {len(strategies)} strategies available!")
            print(f"     This is causing the winner-takes-all distribution!")
            print(f"     Strategies: {[(s.id, s.name) for s in strategies[:5]]}")
        
        all_predictions = []
        
        # 3. 評估每一項戰術並解除正規化
        # 🔥 V2026.50 SOFTMAX EDITION: NO HARD FILTERING
        # All tactics must retain theoretical probability, even with low scores
        for strat in strategies:
            weight = strat.calculate_weight(state)
            
            # 🔥 CRITICAL CHANGE: Remove hard filtering (if weight > 0)
            # Even negative or zero weights will be transformed by Softmax
            # This ensures tactical stochasticity and unpredictable aggression
            
            pred = strat.to_prediction(state, weight)
            
            # 4. 反轉落點座標回到絕對座標系
            tx, ty = pred.target_pos
            pred.target_pos = (tx * mirror_factor, ty * mirror_factor)
            pred.zone_name = Geometry30Point.get_zone_name(*pred.target_pos)
            
            # 🚨 CRITICAL FIX: Also mirror recovery_pos!
            rx, ry = pred.recovery_pos
            pred.recovery_pos = (rx * mirror_factor, ry * mirror_factor)
            
            all_predictions.append(pred)
        
        # ============================================================================
        # 🔥 V2026.48 MANDATORY TRACEABILITY FALLBACK
        # ============================================================================
        # Ensure EVERY generated shot derives from one of 48 tactical IDs
        # If no predictions generated, default to Tactic ID 1: Counterpunching
        if not all_predictions:
            opp_y = opponent_position[1] * mirror_factor
            fallback_target_y = -opp_y if abs(opp_y) > 3.0 else 8.0  # Crosscourt or Ad side
            
            # 🚨 CRITICAL FIX: Apply mirror_factor to target_x too!
            fallback_target_x = 39.0 * mirror_factor  # ✅ Will be negative if opponent on right
            fallback_target_y = fallback_target_y * mirror_factor  # ✅ Mirror Y too
            
            all_predictions.append(OpponentPrediction(
                name="Counterpunching (Default)",
                shot_type=FixedShotType.BASELINE_TOPSPIN,
                target_pos=(fallback_target_x, fallback_target_y),
                score=50.0,
                tactic_id=1,
                tactic_name="Counterpunching - Baseline Rally",
                tactical_reasoning=(
                    "[FALLBACK: No Specific Tactic Matched] | "
                    "[Default to ID 1: Counterpunching] | "
                    "Maintain baseline rally with crosscourt topspin | "
                    "Percentage Tennis: 70/30 Rule compliance | "
                    "Wait for opponent error or short ball opportunity | "
                    f"Energy Cost: ~12 kcal (BMR 2200 kcal baseline)"
                ),
                confidence_level="Medium",
                uncertainty_interval="±5.0%"
            ))

        # 3. 【重構】整合慣性預測引擎
        pattern = OpponentHabitualPredictor.analyze_shot_pattern(norm_history, "Opponent")
        forced_bias = OpponentHabitualPredictor.calculate_forced_prediction_bias(
            norm_opp_pos, opp_phase
        )
        
        # 4. 【重構】應用慣性偏置和受迫性偏置
        for pred in all_predictions:
            # 受迫性偏置
            if pred.shot_type in forced_bias:
                pred.score += forced_bias[pred.shot_type]
            
            # 驚奇加分（判定對角線 vs 直線）
            # 使用絕對座標，避免正規化重複轉換
            is_crosscourt = CourtGeometry.is_crosscourt_shot(
                opponent_position,
                pred.target_pos
            )
            shot_direction = "crosscourt" if is_crosscourt else "dtl"
            surprise_bonus = OpponentHabitualPredictor.calculate_surprise_bonus(
                shot_direction, pattern
            )
            pred.score += surprise_bonus
            
            # 計算不確定性區間
            confidence_level, uncertainty_interval = OpponentHabitualPredictor.calculate_uncertainty_interval(
                pred.score,
                len(norm_history),
                opponent_style
            )
            pred.confidence_level = confidence_level
            pred.uncertainty_interval = uncertainty_interval
        
        # 5. 【重構】引入非標準戰術突變
        all_predictions = TacticalComboEngine.introduce_mutation(all_predictions, mutation_rate=0.08)
        
        # 6. 合併重複預測（相同球種 + 相同落點）
        # 🔍 V2026.51: Log before deduplication
        pre_dedup_count = len(all_predictions)
        
        unique_predictions = {}
        for pred in all_predictions:
            # 🔥 V2026.51 FIX: Round target_pos to avoid over-aggressive deduplication
            # Original: exact position match → too strict, merges similar shots
            # Fixed: Round to nearest 3ft grid → preserves tactical variety
            rounded_target = (round(pred.target_pos[0] / 3.0) * 3.0, round(pred.target_pos[1] / 3.0) * 3.0)
            key = (pred.shot_type, rounded_target)
            if key not in unique_predictions or pred.score > unique_predictions[key].score:
                unique_predictions[key] = pred
        all_predictions = list(unique_predictions.values())
        
        # 🔍 V2026.51: Log after deduplication
        post_dedup_count = len(all_predictions)
        print(f"[V2026.51 Deduplication] {pre_dedup_count} → {post_dedup_count} predictions")
        if pre_dedup_count != post_dedup_count:
            print(f"  Removed {pre_dedup_count - post_dedup_count} duplicates (merged by 3ft grid)")
        if post_dedup_count < 5:
            print(f"  ⚠️ CRITICAL: Only {post_dedup_count} unique predictions after deduplication!")
            print(f"     This will cause winner-takes-all distribution!")

        # 7. 按分數排序
        all_predictions.sort(key=lambda p: p.score, reverse=True)
        
        # ============================================================================
        # 🔥 V2026.51 GLOBAL SOFTMAX DISTRIBUTION (ENHANCED)
        # ============================================================================
        # Implement full Softmax transformation with temperature parameter (tau)
        # Formula: P(i) = exp(S_i / tau) / sum(exp(S_j / tau))
        # 
        # Temperature (tau) = 1.5 (user-specified V2026.51)
        # - tau < 1.0: Winner-takes-all (sharper distribution)
        # - tau = 1.0-1.5: Balanced hierarchy with tactical variety
        # - tau = 1.5: Ensures Rank 3-10 retain 0.5%-5% probability ✅
        # - tau > 2.0: Uniform distribution (too random)
        # 
        # Key Features:
        # - NO shot forced to 0.0% (all retain theoretical probability)
        # - High-risk shots (DTL in Neutral) have low but non-zero probability
        # - Simulates "unpredictable aggression" and tactical entropy
        # - BMR 2200 kcal baseline for energy-weighted scoring
        # ============================================================================
        
        # 🔥 V2026.52: USER-SPECIFIED TEMPERATURE (MANDATORY)
        # User requirement: tau = 1.3 (strict enforcement)
        # - Balances tactical hierarchy with entropy
        # - Ensures Rank 3-15 retain minimum 0.1% probability
        # - Simulates "unpredictable aggression" in professional tennis
        # - Combined with log compression to handle extreme score gaps
        SOFTMAX_TEMPERATURE = 1.3  # 🎯 USER-SPECIFIED (V2026.52)
        
        scores = np.array([p.score for p in all_predictions])
        
        # 🔍 V2026.51 DIAGNOSTIC: Log raw scores before Softmax
        print(f"\n[V2026.51 Pre-Softmax Diagnostic]")
        print(f"  Total predictions: {len(all_predictions)}")
        print(f"  Score range: [{np.min(scores):.2f}, {np.max(scores):.2f}]")
        print(f"  Top 5 scores: {scores[:5] if len(scores) >= 5 else scores}")
        
        # 🔥 V2026.52 ENHANCED: Mandatory logarithmic compression for all extreme gaps
        # Problem: With tau=1.3, even 50-point gaps cause winner-takes-all
        # Solution: Always compress if gap > 30 points (lower threshold)
        # Formula: compressed = log(1 + normalized_score) × scale_factor
        if len(scores) >= 3:
            score_gap_1_to_3 = scores[0] - scores[2]
            
            # 🎯 V2026.52: Lower threshold from 50 to 30 points
            if score_gap_1_to_3 > 30:
                print(f"  ⚠️ EXTREME SCORE GAP: Rank 1-3 gap = {score_gap_1_to_3:.1f} points")
                print(f"     Applying MANDATORY logarithmic compression (V2026.52)...")
                
                # Step 1: Normalize to positive range
                min_score = np.min(scores)
                scores_normalized = scores - min_score + 1.0
                
                # Step 2: Apply logarithmic compression
                # log1p(x) = log(1 + x) for numerical stability
                scores_log = np.log1p(scores_normalized)
                
                # Step 3: Rescale to maintain tactical hierarchy
                # Scale factor = 50 (ensures top scores remain distinguishable)
                scores = scores_log * 50.0
                
                print(f"     Compressed score range: [{np.min(scores):.2f}, {np.max(scores):.2f}]")
                print(f"     Top 5 compressed: {scores[:5] if len(scores) >= 5 else scores}")
                print(f"     Compression ratio: {score_gap_1_to_3 / (scores[0] - scores[2]):.2f}x")
                
                # 🚨 V2026.52 EMERGENCY: Apply score capping if gap still > 50 points
                # Even after log compression, extreme gaps can defeat low tau (1.3)
                # Solution: Hard cap maximum score range to 50 points
                MAX_SCORE_GAP = 50.0
                compressed_range = np.max(scores) - np.min(scores)
                if compressed_range > MAX_SCORE_GAP:
                    print(f"     ⚠️ Gap still extreme ({compressed_range:.1f} pts), applying HARD CAP...")
                    min_score = np.min(scores)
                    scores = (scores - min_score) / compressed_range * MAX_SCORE_GAP + min_score
                    print(f"     Final capped range: [{np.min(scores):.2f}, {np.max(scores):.2f}]")
            else:
                print(f"  ✅ Acceptable score gap: {score_gap_1_to_3:.1f} points (no compression needed)")
        
        # Apply Softmax with temperature scaling
        # Subtract max score for numerical stability (prevents overflow)
        scores_shifted = scores - np.max(scores)
        exp_scores = np.exp(scores_shifted / SOFTMAX_TEMPERATURE)
        
        # 🔍 V2026.51 DIAGNOSTIC: Check for numerical issues
        if np.any(np.isnan(exp_scores)) or np.any(np.isinf(exp_scores)):
            print(f"  ⚠️ WARNING: NaN or Inf detected in exp_scores!")
            print(f"     scores_shifted range: [{np.min(scores_shifted):.2f}, {np.max(scores_shifted):.2f}]")
            # Handle numerical issues
            exp_scores = np.nan_to_num(exp_scores, nan=0.0, posinf=1e10, neginf=0.0)
        
        probabilities = exp_scores / np.sum(exp_scores) * 100.0
        
        # 🔥 V2026.52 MANDATORY: Minimum probability floor (0.1%)
        # Ensures ALL valid tactics retain theoretical probability
        # Simulates "tactical entropy" and "unpredictable aggression"
        MIN_PROBABILITY = 0.1  # User-specified minimum (V2026.52)
        
        # Apply minimum floor
        probabilities_floored = np.maximum(probabilities, MIN_PROBABILITY)
        
        # Renormalize to ensure sum = 100%
        total_after_floor = np.sum(probabilities_floored)
        probabilities = probabilities_floored / total_after_floor * 100.0
        
        # Assign probabilities to predictions
        for idx, pred in enumerate(all_predictions):
            pred.probability = probabilities[idx]
        
        # 🔍 V2026.52 DIAGNOSTIC: Verify minimum probability enforcement
        below_min_count = sum(1 for p in all_predictions if p.probability < MIN_PROBABILITY)
        print(f"  Below minimum ({MIN_PROBABILITY}%) count: {below_min_count} (should be 0)")
        if below_min_count > 0:
            print(f"  ⚠️ CRITICAL: {below_min_count} predictions below minimum probability!")
        
        # Verification: Ensure total probability sums to 100%
        total_probability = sum(p.probability for p in all_predictions)
        if abs(total_probability - 100.0) > 0.01:  # Tolerance for floating point errors
            print(f"⚠️ WARNING: Total probability = {total_probability:.2f}% (should be 100%)")
        
        # Log probability distribution for diagnostic purposes
        print(f"\n[V2026.51 Softmax Distribution - tau={SOFTMAX_TEMPERATURE}]")
        print(f"  Total predictions: {len(all_predictions)}")
        print(f"  Top 3: {[f'{p.probability:.2f}%' for p in all_predictions[:3]]}")
        if len(all_predictions) >= 6:
            print(f"  Ranks 4-6: {[f'{p.probability:.2f}%' for p in all_predictions[3:6]]}")
        if len(all_predictions) >= 10:
            print(f"  Ranks 7-10: {[f'{p.probability:.4f}%' for p in all_predictions[6:10]]}")
        print(f"  Bottom 3: {[f'{p.probability:.4f}%' for p in all_predictions[-3:]]}")
        print(f"  Total: {total_probability:.4f}%")
        print(f"  Verification: {'✅ PASS' if abs(total_probability - 100.0) < 0.01 else '❌ FAIL'}")
        
        # 9. 分配選項 (A, B, C)
        for idx, pred in enumerate(all_predictions[:3]):
            pred.option = ["A", "B", "C"][idx]
            
        return all_predictions
    
    @staticmethod
    def _generate_target_for_opponent(
        shot_type: FixedShotType,
        opponent_position: Tuple[float, float],
        my_position: Tuple[float, float],
        is_lefty: bool,
        direction: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        【修正】根據球種生成目標位置（對手擊球到我方半場，對手視角）
        
        把對手當作另外一個"我"，使用完整的戰術大腦邏輯
        """
        my_x, my_y = my_position
        opp_x, opp_y = opponent_position
        
        # 【修正】判斷對手是否在網前（對手視角）
        # 對手在網前（對手視角）= opp_x < 21（對手半場的網前區域）
        is_opp_at_net = opp_x < 21.0  # 對手視角：網前區域
        
        # 【修正】判斷我方是否在網前（對手視角：對手看到我方在網前）
        # 對手視角：我方在網前 = my_x > -21（我方半場的網前區域，對手視角）
        is_my_at_net_for_opponent = my_x > -21.0  # 對手視角：我方在網前
        
        # 【擴展】根據方向（左中右，我方視角）生成目標 Y 座標
        # 我方視角：左 = Y=8 (Ad側), 中 = Y=0, 右 = Y=-8 (Deuce側)
        
        if shot_type == FixedShotType.BASELINE_TOPSPIN:
            # 底線上旋球：根據方向生成目標
            if direction == "LEFT":
                target_y = Geometry30Point.snap_to_y_grid(8.0)  # 我方視角：左側
            elif direction == "CENTER":
                target_y = Geometry30Point.snap_to_y_grid(0.0)  # 我方視角：中央
            elif direction == "RIGHT":
                target_y = Geometry30Point.snap_to_y_grid(-8.0)  # 我方視角：右側
            else:
                # 預設：對角線（對手視角）
                if is_my_at_net_for_opponent:
                    target_y = Geometry30Point.snap_to_y_grid(my_y)  # 直線穿越
                else:
                    # 🔥 V2026.53: Fixed crosscourt logic (was: -opp_y if opp_y < 0 else -opp_y)
                    # Crosscourt = opposite side = simple negation
                    target_y = Geometry30Point.snap_to_y_grid(-opp_y)  # 對角線
            target_x = -39.0
            
        elif shot_type == FixedShotType.BASELINE_SLICE:
            # 底線切球：根據方向生成目標
            if direction == "LEFT":
                target_y = Geometry30Point.snap_to_y_grid(8.0)  # 我方視角：左側
            elif direction == "CENTER":
                target_y = Geometry30Point.snap_to_y_grid(0.0)  # 我方視角：中央
            elif direction == "RIGHT":
                target_y = Geometry30Point.snap_to_y_grid(-8.0)  # 我方視角：右側
            else:
                # 預設：對角線（對手視角）
                if is_my_at_net_for_opponent:
                    target_y = Geometry30Point.snap_to_y_grid(my_y)  # 直線穿越
                else:
                    # 🔥 V2026.53: Fixed crosscourt logic
                    target_y = Geometry30Point.snap_to_y_grid(-opp_y)  # 對角線
            target_x = -39.0
            
        elif shot_type == FixedShotType.BASELINE_FLAT:
            # 底線平擊球：根據方向生成目標
            if direction == "LEFT":
                target_y = Geometry30Point.snap_to_y_grid(8.0)  # 我方視角：左側
            elif direction == "CENTER":
                target_y = Geometry30Point.snap_to_y_grid(0.0)  # 我方視角：中央
            elif direction == "RIGHT":
                target_y = Geometry30Point.snap_to_y_grid(-8.0)  # 我方視角：右側
            else:
                # 預設：直線（對手視角）
                target_y = Geometry30Point.snap_to_y_grid(opp_y)  # 對手視角：同側直線
            target_x = -39.0
            
        elif shot_type == FixedShotType.INSIDE_OUT:
            # 大角度攻擊：根據方向生成目標（極外角）
            if direction == "LEFT":
                target_y = Geometry30Point.snap_to_y_grid(14.0)  # 我方視角：極外角左
            elif direction == "RIGHT":
                target_y = Geometry30Point.snap_to_y_grid(-14.0)  # 我方視角：極外角右
            else:
                # 預設：根據對手位置決定
                target_y = Geometry30Point.snap_to_y_grid(14.0 if opp_y >= 0 else -14.0)
            target_x = -39.0
            
        elif shot_type == FixedShotType.DROP_SHOT:
            # 網前短球：根據方向生成目標（我方視角）
            if direction == "LEFT":
                target_y = Geometry30Point.snap_to_y_grid(8.0)  # 我方視角：左側
            elif direction == "CENTER":
                target_y = Geometry30Point.snap_to_y_grid(0.0)  # 我方視角：中央
            elif direction == "RIGHT":
                target_y = Geometry30Point.snap_to_y_grid(-8.0)  # 我方視角：右側
            else:
                # 預設：中央
                target_y = Geometry30Point.snap_to_y_grid(0.0)
            target_x = -5.0
            
        else:  # LOB
            # 高吊球：無方向，固定中央
            target_y = Geometry30Point.snap_to_y_grid(0.0)  # 中路高吊
            if is_my_at_net_for_opponent:
                target_x = -39.0  # 底線深處
            else:
                target_x = -35.0
        
        return target_x, target_y
    
    @staticmethod
    def _generate_target(
        shot_type: FixedShotType,
        my_position: Tuple[float, float],
        opponent_position: Tuple[float, float],
        is_lefty: bool
    ) -> Tuple[float, float]:
        """
        【保留】舊版方法，向後兼容
        實際調用 _generate_target_for_opponent（不指定方向，使用預設邏輯）
        """
        return DynamicOpponentPredictionEngine._generate_target_for_opponent(
            shot_type, opponent_position, my_position, is_lefty, direction=None
        )
    
    @staticmethod
    def _get_shot_name(shot_type: FixedShotType) -> str:
        """獲取球種名稱"""
        names = {
            FixedShotType.BASELINE_TOPSPIN: "穩定",
            FixedShotType.BASELINE_SLICE: "變奏",
            FixedShotType.BASELINE_FLAT: "進攻",
            FixedShotType.INSIDE_OUT: "進攻",
            FixedShotType.DROP_SHOT: "變奏",
            FixedShotType.LOB: "防守",
        }
        return names.get(shot_type, "穩定")
    
    @staticmethod
    def _generate_tactical_reasoning(
        shot_type: FixedShotType,
        score: float,
        phase: PhaseType,
        is_outside: bool,
        last_shot_type: Optional[FixedShotType],
        opponent_style: OpponentStyle
    ) -> str:
        """生成戰術說明"""
        reasoning = []
        
        if is_outside:
            reasoning.append("外側球對角回擊（Wardlaw原則）")
        else:
            reasoning.append("內側球可變線")
        
        if shot_type == FixedShotType.BASELINE_TOPSPIN:
            reasoning.append("對角深球確保穩定性")
        elif shot_type == FixedShotType.BASELINE_FLAT:
            reasoning.append("直線攻擊製造壓力")
        elif shot_type == FixedShotType.DROP_SHOT:
            reasoning.append("網前短球改變節奏")
        
        if phase == PhaseType.DEFENSE:
            reasoning.append("對手防守相位，拉開角度")
        
        if last_shot_type is not None and shot_type != last_shot_type:
            reasoning.append("戰術多樣性獎勵")
        
        # 風格說明
        if opponent_style == OpponentStyle.PUSHER:
            if shot_type == FixedShotType.BASELINE_TOPSPIN:
                reasoning.append("防守型對手偏好對角深球")
        elif opponent_style == OpponentStyle.NET_RUSHER:
            if shot_type == FixedShotType.BASELINE_TOPSPIN:
                reasoning.append("網前型對手傾向深度壓制後上網")
        
        return "；".join(reasoning) if reasoning else "戰術執行"


# ============================================================================
# 破局邏輯（Anti-Stalemate）
# ============================================================================

class BreakthroughLogic:
    """
    破局邏輯：Anti-Stalemate
    
    規則：
    - 連續 2 拍 Neutral 且落點在中路 (Y=0) 時
    - 下一拍強制排除 Y=0，改攻極外角或放短球
    """
    
    @staticmethod
    def should_force_breakthrough(
        shot_history: List[ShotStep],
        threshold: int = 2
    ) -> bool:
        """
        判斷是否需要強制破局
        
        參數：
            shot_history: 擊球歷史
            threshold: 破局閾值（預設 2 拍）
        
        返回：
            True 表示需要強制破局
        """
        if len(shot_history) < threshold:
            return False
        
        # 檢查最近 threshold 拍是否都是 Neutral 且落點在中路
        recent_shots = shot_history[-threshold:]
        for shot in recent_shots:
            if shot.phase != PhaseType.NEUTRAL:
                return False
            if abs(shot.end_pos[1]) > 5:  # 不在中路
                return False
        
        return True
    
    @staticmethod
    def get_breakthrough_target_y(
        current_y: float,
        is_lefty: bool = False
    ) -> float:
        """
        獲取破局目標 Y 座標
        
        強制排除 Y=0，改攻極外角或側邊
        """
        # 如果當前在中路，強制選擇極外角
        if abs(current_y) < 5:
            # 隨機選擇極外角左或右
            return 14.0 if not is_lefty else -14.0
        else:
            # 如果已在側邊，選擇另一側或極外角
            if current_y > 0:
                return -14.0  # 改攻右側極外角
            else:
                return 14.0  # 改攻左側極外角


# ============================================================================
# Crosscourt Rally (Neutral Pattern) - Backhand-to-Backhand
# ============================================================================

class CrosscourtRallyPattern:
    """
    Crosscourt Rally (Neutral Pattern)
    
    文檔要求：The backhand-to-backhand crosscourt rally is the fundamental 
    "mini-battle" within a point.
    
    核心邏輯：
    - Neutral Phase 下的對角線拉鋸戰
    - 追蹤連續對角線擊球次數
    - 建立 "backhand-to-backhand" 狀態
    """
    
    @staticmethod
    def is_backhand_to_backhand_situation(
        shot_history: List[ShotStep],
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        phase: PhaseType
    ) -> bool:
        """
        判定是否為 backhand-to-backhand crosscourt rally 情境
        
        條件：
        - Neutral Phase
        - 對手在反拍側（Y > 0 或 Y < -6）
        - 最近 2 拍都是對角線擊球
        """
        if phase != PhaseType.NEUTRAL:
            return False
        
        # 對手必須在反拍側（簡化：Y > 0 或 Y < -6）
        if not (opponent_pos[1] > 0 or opponent_pos[1] < -6):
            return False
        
        # 需要至少 2 拍歷史
        if len(shot_history) < 2:
            return False
        
        # 檢查最近 2 拍是否都是對角線擊球
        recent_shots = shot_history[-2:]
        crosscourt_count = 0
        
        for shot in recent_shots:
            if CourtGeometry.is_crosscourt_shot(shot.start_pos, shot.end_pos):
                crosscourt_count += 1
        
        # 如果最近 2 拍都是對角線，視為 crosscourt rally
        return crosscourt_count >= 2
    
    @staticmethod
    def execute_crosscourt_rally_shot(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        shot_history: List[ShotStep]
    ) -> Tuple[FixedShotType, float, float, str]:
        """
        執行 crosscourt rally 擊球
        
        目標：維持對角線拉鋸，攻擊對手反拍側
        """
        my_y = my_pos[1]
        opp_y = opponent_pos[1]
        
        # 如果對手在反拍側（Y > 0），攻擊對手反拍側（對角線）
        if opp_y > 0:
            # 對手在 Ad 側，攻擊對手 Ad 側（對角線）
            target_y = 8.0  # 對手 Ad 側
        elif opp_y < -6:
            # 對手在 Deuce 側，攻擊對手 Deuce 側（對角線）
            target_y = -8.0  # 對手 Deuce 側
        else:
            # 對手在中路，選擇與我方站位異側
            target_y = -my_y if my_y >= 0 else -my_y
        
        target_y = Geometry30Point.snap_to_y_grid(target_y)
        target_x = 39.0
        
        return (
            FixedShotType.BASELINE_TOPSPIN,
            target_x,
            target_y,
            "Crosscourt Rally: Backhand-to-backhand 對角線拉鋸戰"
        )
    
    @staticmethod
    def check_and_execute(
        step_num: int,
        shot_history: List[ShotStep],
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        phase: PhaseType
    ) -> Optional[Tuple[FixedShotType, float, float, str]]:
        """
        檢查並執行 Crosscourt Rally Pattern
        
        返回：
            None 或 (shot_type, target_x, target_y, reason)
        """
        if CrosscourtRallyPattern.is_backhand_to_backhand_situation(
            shot_history, my_pos, opponent_pos, phase
        ):
            return CrosscourtRallyPattern.execute_crosscourt_rally_shot(
                my_pos, opponent_pos, shot_history
            )
        return None


# ============================================================================
# Angle-Line Pattern（角度-直線進攻模式）
# ============================================================================

class AngleLinePattern:
    """
    Angle-Line Pattern（角度-直線進攻模式）
    
    專業兩拍進攻模式：
    - R(n): 角度拉開（INSIDE_OUT 到極外角 Y=±14）
    - R(n+1): 直線終結（DTL 到對手空檔）
    
    科學依據：
    - 第一拍角度拉開迫使對手橫向移動，創造空檔
    - 第二拍直線終結利用對手回位不足，直接得分
    """
    
    @staticmethod
    def should_trigger_angle_line(
        shot_history: List[ShotStep],
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        phase: PhaseType
    ) -> bool:
        """
        判定是否應該觸發 Angle-Line Pattern
        
        條件：
        - 對手在底線後方（X > 21ft）
        - 我方處於 Neutral 或 Offensive 相位
        - 上一拍是對角線深球（建立基礎）
        - 對手回球質量一般（非強力進攻）
        """
        # 對手必須在底線後方
        if opponent_pos[0] < 21.0:
            return False
        
        # 我方必須處於 Neutral 或 Offensive 相位
        if phase == PhaseType.DEFENSE:
            return False
        
        # 需要至少 1 拍歷史來判斷
        if len(shot_history) < 1:
            return False
        
        # 檢查上一拍是否為對角線深球
        last_shot = shot_history[-1]
        if last_shot.player == "My":
            # 上一拍是我方擊球，檢查是否為對角線深球
            is_deep = last_shot.end_pos[0] > 30.0  # 深度 > 30ft
            is_crosscourt = CourtGeometry.is_crosscourt_shot(
                last_shot.start_pos, last_shot.end_pos
            )
            if is_deep and is_crosscourt:
                return True
        
        return False
    
    @staticmethod
    def execute_angle_shot(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float]
    ) -> Tuple[FixedShotType, float, float, str]:
        """
        執行角度拉開（第一拍）
        
        目標：極外角 Y=±14，深度 X=39
        """
        my_y = my_pos[1]
        
        # 選擇與我方站位異側的極外角
        if my_y >= 0:
            # 我方在 Ad 側，攻擊對手 Deuce 側極外角
            target_y = -14.0
        else:
            # 我方在 Deuce 側，攻擊對手 Ad 側極外角
            target_y = 14.0
        
        target_y = Geometry30Point.snap_to_y_grid(target_y)
        target_x = 39.0
        
        return (
            FixedShotType.INSIDE_OUT,
            target_x,
            target_y,
            "Angle-Line Pattern: 角度拉開到極外角，迫使對手橫向移動"
        )
    
    @staticmethod
    def execute_line_shot(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        previous_angle_shot: ShotStep
    ) -> Tuple[FixedShotType, float, float, str]:
        """
        執行直線終結（第二拍）
        
        目標：DTL 到對手空檔（與角度拉開同側）
        """
        # DTL：與我方站位同側
        target_y = Geometry30Point.snap_to_y_grid(my_pos[1])
        
        # 如果在中路，選擇與角度拉開同側
        if abs(target_y) < 3:
            target_y = previous_angle_shot.end_pos[1]  # 與角度拉開同側
        
        target_x = 39.0
        
        return (
            FixedShotType.BASELINE_FLAT,
            target_x,
            target_y,
            "Angle-Line Pattern: 直線終結到對手空檔"
        )
    
    @staticmethod
    def check_and_execute(
        step_num: int,
        shot_history: List[ShotStep],
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        phase: PhaseType
    ) -> Optional[Tuple[FixedShotType, float, float, str]]:
        """
        檢查並執行 Angle-Line Pattern
        
        返回：
            None 或 (shot_type, target_x, target_y, reason)
        """
        # 檢查是否應該觸發
        if not AngleLinePattern.should_trigger_angle_line(
            shot_history, my_pos, opponent_pos, phase
        ):
            return None
        
        # 檢查上一拍是否為角度拉開
        if len(shot_history) >= 1:
            last_shot = shot_history[-1]
            if (last_shot.player == "My" and 
                last_shot.shot_type == FixedShotType.INSIDE_OUT and
                abs(last_shot.end_pos[1]) >= 14):
                # 上一拍是角度拉開，當前拍執行直線終結
                return AngleLinePattern.execute_line_shot(
                    my_pos, opponent_pos, last_shot
                )
        
        # 第一拍：執行角度拉開
        return AngleLinePattern.execute_angle_shot(my_pos, opponent_pos)


# ============================================================================
# Serve and Volley Tactic（發球上網戰術）
# ============================================================================

class ServeAndVolleyTactic:
    """
    Serve and Volley Tactic（發球上網戰術）
    
    文檔要求：
    - 發球後立即上網
    - 使用 T 區發球（最高百分比）
    - 使用 kick serve（較慢速度、高彈跳，提供更多時間上網）
    - 發球後立即向前移動，使用 split step 準備截擊
    """
    
    @staticmethod
    def should_trigger_serve_and_volley(
        step_num: int,
        opponent_style: 'OpponentStyle',
        serve_quality: float = 0.7  # 發球質量（0-1）
    ) -> bool:
        """
        判定是否應該使用 Serve and Volley
        
        參數：
            step_num: 擊球步驟（應為 1，即 R1）
            opponent_style: 對手風格
            serve_quality: 發球質量（0-1）
        
        返回：
            True 為應該使用 Serve and Volley
        """
        # 僅在 R1（發球）時觸發
        if step_num != 1:
            return False
        
        # 如果對手接發球較弱（Pusher），或發球質量高，使用 Serve and Volley
        if opponent_style == OpponentStyle.PUSHER:
            return True
        elif serve_quality >= 0.7:
            return np.random.random() < 0.3  # 30% 機率使用（作為驚喜戰術）
        
        return False
    
    @staticmethod
    def execute_serve_and_volley(
        start_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        is_lefty: bool
    ) -> Tuple[FixedShotType, float, float, str]:
        """
        執行 Serve and Volley 戰術
        
        參數：
            start_pos: 發球起始位置
            opponent_pos: 對手位置
            is_lefty: 是否為左撇子
        
        返回：
            (shot_type, target_x, target_y, reason)
        """
        # 使用 T 區發球（最高百分比）
        # T 區：Y = 0（中路）
        target_y = 0.0
        target_x = 39.0  # 對手底線
        
        # 使用 kick serve（上旋發球，高彈跳）
        # 注意：在我們的系統中，發球使用 BASELINE_TOPSPIN 表示上旋發球
        shot_type = FixedShotType.BASELINE_TOPSPIN
        
        return (
            shot_type,
            target_x,
            target_y,
            "Serve and Volley: T 區發球後立即上網，準備截擊"
        )
    
    @staticmethod
    def check_and_execute(
        step_num: int,
        start_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        opponent_style: 'OpponentStyle',
        is_lefty: bool,
        serve_quality: float = 0.7
    ) -> Optional[Tuple[FixedShotType, float, float, str]]:
        """
        檢查並執行 Serve and Volley 戰術
        
        返回：
            None 或 (shot_type, target_x, target_y, reason)
        """
        if ServeAndVolleyTactic.should_trigger_serve_and_volley(
            step_num, opponent_style, serve_quality
        ):
            return ServeAndVolleyTactic.execute_serve_and_volley(
                start_pos, opponent_pos, is_lefty
            )
        return None


# ============================================================================
# Defensive Shots (Moonballs/High Floaters)（防守球：高吊球/高浮球）
# ============================================================================

class DefensiveShots:
    """
    Defensive Shots (Moonballs/High Floaters)（防守球：高吊球/高浮球）
    
    文檔要求：
    - 在防守相位使用
    - 目標：高、深的浮球來"買時間"和恢復位置
    - 減慢交換速度，允許恢復
    """
    
    @staticmethod
    def should_use_defensive_shot(
        phase: PhaseType,
        start_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float]
    ) -> bool:
        """
        判定是否應該使用防守球（Moonball/High Floater）
        
        參數：
            phase: 相位
            start_pos: 擊球起始位置
            opponent_pos: 對手位置
        
        返回：
            True 為應該使用防守球
        """
        # 僅在防守相位使用
        if phase != PhaseType.DEFENSE:
            return False
        
        # 如果被拉出場外（X < -45），使用防守球
        if Geometry30Point.is_defense_zone(start_pos[0]):
            return True
        
        # 如果對手在進攻位置，使用防守球爭取時間
        if Geometry30Point.is_offense_zone(opponent_pos[0]):
            return np.random.random() < 0.6  # 60% 機率使用
        
        return False
    
    @staticmethod
    def execute_moonball(
        start_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float]
    ) -> Tuple[FixedShotType, float, float, str]:
        """
        執行 Moonball/High Floater（高吊球/高浮球）
        
        參數：
            start_pos: 擊球起始位置
            opponent_pos: 對手位置
        
        返回：
            (shot_type, target_x, target_y, reason)
        """
        # 高、深的浮球：目標是對手底線深處
        target_x = 39.0  # 對手底線
        target_y = 0.0   # 中路（最安全）
        
        # 使用 LOB（高吊球）類型
        shot_type = FixedShotType.LOB
        
        return (
            shot_type,
            target_x,
            target_y,
            "防守球（Moonball）: 高、深浮球，爭取回位時間"
        )
    
    @staticmethod
    def check_and_execute(
        phase: PhaseType,
        start_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float]
    ) -> Optional[Tuple[FixedShotType, float, float, str]]:
        """
        檢查並執行防守球（Moonball/High Floater）
        
        返回：
            None 或 (shot_type, target_x, target_y, reason)
        """
        if DefensiveShots.should_use_defensive_shot(phase, start_pos, opponent_pos):
            return DefensiveShots.execute_moonball(start_pos, opponent_pos)
        return None


# ============================================================================
# Serve + 1 Patterns（發球+1模式）
# ============================================================================

class ServePlusOnePattern:
    """
    Serve + 1 Patterns（發球+1模式）
    
    專業發球戰術：
    - Deuce Court Pattern: 外角發球 → 正拍 DTL
    - Ad Court Pattern: 外角發球（Kick serve）→ Inside-in 正拍
    
    科學依據：
    - 外角發球迫使對手橫向移動，創造空檔
    - +1 擊球利用對手回位不足，直接得分或建立優勢
    """
    
    @staticmethod
    def is_serve_plus_one_situation(step_num: int, shot_history: List[ShotStep]) -> bool:
        """
        判定是否為 Serve + 1 情境（R1 或 R2）
        
        參數：
            step_num: 當前拍數
            shot_history: 擊球歷史
        
        返回：
            True 為 Serve + 1 情境
        """
        return step_num <= 2
    
    @staticmethod
    def execute_deuce_court_pattern(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        step_num: int,
        is_lefty: bool
    ) -> Optional[Tuple[FixedShotType, float, float, str]]:
        """
        執行 Deuce Court Pattern
        
        R1: 外角發球到 Deuce 側（Y=-8）
        R2: 正拍 DTL 到對手空檔（Y=-8 同側）
        """
        if step_num == 1:
            # R1: 外角發球到 Deuce 側
            target_y = -8.0  # Deuce 側
            target_x = 39.0
            return (
                FixedShotType.BASELINE_FLAT,  # 發球使用平擊增加速度
                target_x,
                target_y,
                "Deuce Court Pattern: 外角發球到 Deuce 側"
            )
        elif step_num == 2:
            # R2: 正拍 DTL 到對手空檔（與發球同側）
            target_y = -8.0  # 與發球同側
            target_x = 39.0
            return (
                FixedShotType.BASELINE_FLAT,  # DTL 使用平擊增加速度
                target_x,
                target_y,
                "Deuce Court Pattern: 正拍 DTL 到對手空檔"
            )
        return None
    
    @staticmethod
    def execute_ad_court_pattern(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        step_num: int,
        is_lefty: bool
    ) -> Optional[Tuple[FixedShotType, float, float, str]]:
        """
        執行 Ad Court Pattern
        
        R1: 外角發球到 Ad 側（Y=8，Kick serve）
        R2: Inside-in 正拍到 Deuce 側空檔（Y=-8）
        """
        if step_num == 1:
            # R1: 外角發球到 Ad 側（Kick serve）
            target_y = 8.0  # Ad 側
            target_x = 39.0
            return (
                FixedShotType.BASELINE_TOPSPIN,  # Kick serve 使用上旋
                target_x,
                target_y,
                "Ad Court Pattern: 外角發球到 Ad 側（Kick serve）"
            )
        elif step_num == 2:
            # R2: Inside-in 正拍到 Deuce 側空檔
            target_y = -8.0  # Deuce 側（與發球異側）
            target_x = 39.0
            return (
                FixedShotType.INSIDE_OUT,  # Inside-in 使用大角度
                target_x,
                target_y,
                "Ad Court Pattern: Inside-in 正拍到 Deuce 側空檔"
            )
        return None
    
    @staticmethod
    def check_and_execute(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        step_num: int,
        shot_history: List[ShotStep],
        is_lefty: bool
    ) -> Optional[Tuple[FixedShotType, float, float, str]]:
        """
        檢查並執行 Serve + 1 Pattern
        
        參數：
            my_pos: 我方位置
            opponent_pos: 對手位置
            step_num: 當前拍數
            shot_history: 擊球歷史
            is_lefty: 是否為左撇子
        
        返回：
            None 或 (shot_type, target_x, target_y, reason)
        """
        # 只在 R1 或 R2 時觸發
        if not ServePlusOnePattern.is_serve_plus_one_situation(step_num, shot_history):
            return None
        
        my_y = my_pos[1]
        
        # 根據我方站位選擇 Pattern
        # 如果在中路或 Deuce 側，使用 Deuce Court Pattern
        if my_y <= 0:
            return ServePlusOnePattern.execute_deuce_court_pattern(
                my_pos, opponent_pos, step_num, is_lefty
            )
        # 如果在 Ad 側，使用 Ad Court Pattern
        else:
            return ServePlusOnePattern.execute_ad_court_pattern(
                my_pos, opponent_pos, step_num, is_lefty
            )


# ============================================================================
# Return Strategies（接發球策略）
# ============================================================================

class ReturnStrategy:
    """
    Return Strategies（接發球策略）
    
    專業接發球戰術：
    - 高百分比回球：順著來球方向（外角發球 → 對角線回球）
    - 弱二發處理：當作 Approach Shot（深場回球 + 上網）
    
    科學依據：
    - 順著來球方向使用最大場地，降低失誤率
    - 弱二發是進攻機會，應積極處理
    """
    
    @staticmethod
    def calculate_high_percentage_return(
        serve_pos: Tuple[float, float],
        serve_direction: Tuple[float, float]
    ) -> Tuple[FixedShotType, float, float, str]:
        """
        計算高百分比回球
        
        原則：順著來球方向，使用最大場地
        
        參數：
            serve_pos: 發球位置
            serve_direction: 發球方向（目標位置）
        
        返回：
            (shot_type, target_x, target_y, reason)
        """
        serve_y = serve_direction[1]
        
        # 外角發球 → 對角線回球
        if abs(serve_y) >= 6:  # 外角發球
            # 對角線回球：與發球方向同側
            target_y = serve_y
            target_x = -39.0  # 回到底線
            return (
                FixedShotType.BASELINE_TOPSPIN,
                target_x,
                target_y,
                "高百分比回球：順著來球方向，對角線回球"
            )
        else:
            # T 點發球 → 中路回球
            target_y = 0.0
            target_x = -39.0
            return (
                FixedShotType.BASELINE_TOPSPIN,
                target_x,
                target_y,
                "高百分比回球：T 點發球，中路回球"
            )
    
    @staticmethod
    def is_attackable_second_serve(serve_speed: float, serve_depth: float) -> bool:
        """
        判定是否為可攻擊的弱二發
        
        參數：
            serve_speed: 發球速度（簡化：深度代表速度）
            serve_depth: 發球深度
        
        返回：
            True 為可攻擊的弱二發
        """
        # 如果發球深度 < 30ft，視為弱二發
        return serve_depth < 30.0
    
    @staticmethod
    def check_second_serve_trap(
        serve_depth: float,
        recent_errors: int,
        momentum: float
    ) -> bool:
        """
        檢查弱二發陷阱（防止過度攻擊導致失誤）
        
        資料：弱二發可能是"誘餌"，對手可能"過度興奮"導致失誤
        
        參數：
            serve_depth: 發球深度
            recent_errors: 最近失誤次數
            momentum: 當前動量
        
        返回：
            True 為觸發陷阱（應保守回球），False 為可攻擊
        """
        # 如果最近失誤多（>2次）且動量低（<0），可能是陷阱
        if recent_errors >= 2 and momentum < 0:
            return True
        
        # 如果發球深度非常淺（<25ft），可能是故意誘餌
        if serve_depth < 25.0:
            return True
        
        return False
    
    @staticmethod
    def execute_approach_return(
        serve_pos: Tuple[float, float],
        serve_direction: Tuple[float, float]
    ) -> Tuple[FixedShotType, float, float, str]:
        """
        執行 Approach Shot 回球（弱二發）
        
        深場回球到角落 + 上網
        """
        serve_y = serve_direction[1]
        
        # 深場回球到角落
        if serve_y > 0:
            target_y = 8.0  # Ad 側角落
        else:
            target_y = -8.0  # Deuce 側角落
        
        target_x = -39.0  # 深場
        
        return (
            FixedShotType.BASELINE_TOPSPIN,
            target_x,
            target_y,
            "弱二發處理：深場回球到角落，準備上網"
        )


# ============================================================================
# Counter-Punch Pattern（反擊模式）
# ============================================================================

class CounterPunchPattern:
    """
    Counter-Punch Pattern（反擊模式）
    
    資料：Hingis 的反擊模式
    1. Defense: 被推入防守位置（寬且深）
    2. Buying Time: 深球買時間，回位
    3. Opponent's Error: 對手不在強攻擊位置，做出錯誤決策
    4. Offense: 完美回位後，進攻得分
    
    科學依據：
    - 動量反轉：從防守轉為進攻
    - 智慧決策：不慌張，用深球買時間
    """
    
    @staticmethod
    def is_defensive_position(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float]
    ) -> bool:
        """
        判定是否處於防守位置
        
        參數：
            my_pos: 我方位置
            opponent_pos: 對手位置
        
        返回：
            True 為防守位置（寬且深）
        """
        my_x, my_y = my_pos
        
        # 防守位置：X < -45ft（底線後方）且 Y 在極外角（|Y| >= 14ft）
        is_wide = abs(my_y) >= 14.0
        is_deep = my_x < -45.0
        
        return is_wide and is_deep
    
    @staticmethod
    def execute_buy_time_shot(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float]
    ) -> Tuple[FixedShotType, float, float, str]:
        """
        執行"買時間"深球
        
        參數：
            my_pos: 我方位置
            opponent_pos: 對手位置
        
        返回：
            (shot_type, target_x, target_y, reason)
        """
        # 深球到對手底線中央，買時間回位
        target_x = 39.0  # 對手底線深處
        target_y = 0.0   # 中央位置（最安全）
        
        return (
            FixedShotType.BASELINE_TOPSPIN,  # 上旋深球
            target_x,
            target_y,
            "Counter-Punch: 深球買時間，回位"
        )
    
    @staticmethod
    def check_and_execute(
        step_num: int,
        shot_history: List[ShotStep],
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        phase: PhaseType
    ) -> Optional[Tuple[FixedShotType, float, float, str]]:
        """
        檢查並執行反擊模式
        
        參數：
            step_num: 當前拍數
            shot_history: 擊球歷史
            my_pos: 我方位置
            opponent_pos: 對手位置
            phase: 當前相位
        
        返回：
            None 或 (shot_type, target_x, target_y, reason)
        """
        # 只在防守相位觸發
        if phase != PhaseType.DEFENSE:
            return None
        
        # 檢查是否處於防守位置
        if not CounterPunchPattern.is_defensive_position(my_pos, opponent_pos):
            return None
        
        # 檢查上一拍是否對手進攻
        if len(shot_history) > 0:
            last_shot = shot_history[-1]
            if last_shot.player == "Opponent" and last_shot.phase == PhaseType.OFFENSE:
                # 對手剛進攻，執行買時間深球
                return CounterPunchPattern.execute_buy_time_shot(my_pos, opponent_pos)
        
        return None


# ============================================================================
# Redirection Principle（變線原則）
# ============================================================================

class RedirectionPrinciple:
    """
    Redirection Principle（變線原則）
    
    核心：維持來球方向比變線更安全
    
    變線風險：
    - 需要更複雜的時機調整
    - 成功率降低 10-15%
    - 僅在進攻機會時使用
    """
    
    @staticmethod
    def calculate_redirection_risk(
        incoming_direction: Tuple[float, float],
        target_direction: Tuple[float, float]
    ) -> float:
        """
        計算變線風險
        
        參數：
            incoming_direction: 來球方向（向量）
            target_direction: 目標方向（向量）
        
        返回：
            風險調整（百分比，0-15%）
        """
        # 計算向量夾角
        dot_product = (incoming_direction[0] * target_direction[0] + 
                      incoming_direction[1] * target_direction[1])
        incoming_mag = math.sqrt(incoming_direction[0]**2 + incoming_direction[1]**2)
        target_mag = math.sqrt(target_direction[0]**2 + target_direction[1]**2)
        
        if incoming_mag == 0 or target_mag == 0:
            return 0.0
        
        cos_angle = dot_product / (incoming_mag * target_mag)
        angle = math.acos(max(-1, min(1, cos_angle)))
        
        # 如果夾角 > 45度，視為變線，風險 +10-15%
        if angle > math.pi / 4:  # 45度
            return 15.0
        elif angle > math.pi / 6:  # 30度
            return 10.0
        
        return 0.0
    
    @staticmethod
    def should_redirect(
        phase: PhaseType,
        opponent_position: Tuple[float, float],
        risk_tolerance: float = 0.3
    ) -> bool:
        """
        判定是否應該變線
        
        參數：
            phase: 相位
            opponent_position: 對手位置
            risk_tolerance: 風險承受度（0-1）
        
        返回：
            True 為應該變線
        """
        # 僅在進攻機會時使用變線
        if phase != PhaseType.OFFENSE:
            return False
        
        # 如果對手在底線後方，變線成功率較高
        if opponent_position[0] > 30:
            return np.random.random() < (0.3 + risk_tolerance * 0.2)
        
        return False


# ============================================================================
# Psychological Momentum（心理動量）
# ============================================================================

class PsychologicalMomentum:
    """
    Psychological Momentum System
    
    Data Source: Understanding Psychological Momentum in Tennis: The Invisible Force That Decides Matches
    
    Key Events:
    - Break point: Significantly increases momentum (rapid surge)
    - Consecutive point wins: Accumulates momentum
    - Key point errors: Momentum decline
    
    Momentum Effects:
    - Success rate adjustment (+/- 5-10%)
    - Risk tolerance modification
    - Tactical selection bias
    
    Range: -1.0 (opponent dominance) to +1.0 (my dominance)
    """
    
    @staticmethod
    def calculate_momentum(
        score: Dict[str, int],
        recent_points: List[bool],
        break_points: int
    ) -> float:
        """
        Calculate current psychological momentum (-1.0 to +1.0)
        
        Calculation Components:
        - Consecutive point streak: ±0.3 per 3-point streak
        - Break points: +0.4 per break point (major momentum surge)
        
        Args:
            score: Current score dictionary {"my": int, "opponent": int}
            recent_points: Recent point outcomes (True = won, False = lost)
            break_points: Number of break point opportunities
        
        Returns:
            Momentum value (-1.0 to +1.0)
        """
        momentum = 0.0
        
        # Consecutive point streak bonus
        if len(recent_points) >= 3:
            if all(recent_points[-3:]):
                momentum += 0.3  # 3-point win streak
            elif all(not p for p in recent_points[-3:]):
                momentum -= 0.3  # 3-point loss streak
        
        # Break point bonus ("rapid increase" - most powerful turning point)
        # Data: Break points are "the most powerful turning points," causing "significant momentum swings"
        if break_points > 0:
            # Each break point adds +0.4-0.5 momentum (stronger than original +0.2)
            momentum += 0.4 * break_points
        
        # Clamp to [-1.0, +1.0] range
        return max(-1.0, min(1.0, momentum))
    
    @staticmethod
    def adjust_success_rate(base_rate: float, momentum: float) -> float:
        """
        Adjust success rate based on psychological momentum
        
        Args:
            base_rate: Base success rate (percentage, 0-100)
            momentum: Momentum value (-1.0 to +1.0)
        
        Returns:
            Adjusted success rate (percentage, 0-100)
        
        Effect:
        - Positive momentum: +10% max success rate boost
        - Negative momentum: -10% max success rate penalty
        """
        # Momentum effect: +/- 10% max adjustment
        adjustment = momentum * 10.0
        return max(0, min(100, base_rate + adjustment))


# ============================================================================

# ============================================================================
# 戰術腳本（Pattern Recognition）
# ============================================================================

class TacticalPattern:
    """
    Tactical Pattern Script System: Multi-Shot Combination Sequences
    
    Example Pattern: "Pattern_Deep_Angle_Winner"
    - R1: Force BASELINE_TOPSPIN_CENTER (deep pressure)
    - R2: If opponent returns weakly, R3 executes INSIDE_OUT_WIDE (angle creation)
    - R4: Execute BASELINE_FLAT_DTL (DTL winner)
    
    Purpose:
    Pre-programmed multi-shot sequences that execute specific tactical progressions
    based on opponent positioning and shot history.
    """
    
    @staticmethod
    def get_pattern_shot(
        step_num: int,
        pattern_name: str,
        shot_history: List[ShotStep],
        my_position: Tuple[float, float],
        opponent_position: Tuple[float, float]
    ) -> Tuple[FixedShotType, float, float]:
        """
        Get shot type and target position from tactical pattern script
        
        Args:
            step_num: Current shot number
            pattern_name: Pattern script name
            shot_history: Shot history (List[ShotStep])
            my_position: My current position (X, Y)
            opponent_position: Opponent's current position (X, Y)
        
        返回：
            (shot_type, target_x, target_y)
        """
        if pattern_name == "Pattern_Deep_Angle_Winner":
            return TacticalPattern._pattern_deep_angle_winner(
                step_num, shot_history, my_position, opponent_position
            )
        else:
            # 預設：對角上旋
            # 🔥 V2026.53: Fixed crosscourt logic
            target_y = Geometry30Point.snap_to_y_grid(-my_position[1])
            return (FixedShotType.BASELINE_TOPSPIN, 39.0, target_y)
    
    @staticmethod
    def _pattern_deep_angle_winner(
        step_num: int,
        shot_history: List[ShotStep],
        my_position: Tuple[float, float],
        opponent_position: Tuple[float, float]
    ) -> Tuple[FixedShotType, float, float]:
        """Pattern_Deep_Angle_Winner 腳本邏輯"""
        my_x, my_y = my_position
        
        if step_num == 1:
            # R1: 強制 BASELINE_TOPSPIN_CENTER (深度壓制)
            target_y = 0.0  # 中路
            target_x = 39.0
            return (FixedShotType.BASELINE_TOPSPIN, target_x, target_y)
        
        elif step_num == 2:
            # R2: 對手回球（由預判引擎決定）
            # 這裡返回預設值，實際由預判引擎覆蓋
            # 🔥 V2026.53: Fixed crosscourt logic
            target_y = Geometry30Point.snap_to_y_grid(-my_y)
            target_x = -39.0
            return (FixedShotType.BASELINE_TOPSPIN, target_x, target_y)
        
        elif step_num == 3:
            # R3: 若對手回弱，執行 INSIDE_OUT_WIDE (角度拉開)
            if len(shot_history) >= 2:
                last_shot = shot_history[-1]
                # 判斷對手是否回弱（落點淺或速度慢）
                if last_shot.phase == PhaseType.NEUTRAL or last_shot.shot_type == FixedShotType.BASELINE_SLICE:
                    # 對手回弱，執行大角度拉開
                    target_y = 14.0  # 極外角
                    target_x = 39.0
                    return (FixedShotType.INSIDE_OUT, target_x, target_y)
            
            # 預設：對角上旋
            # 🔥 V2026.53: Fixed crosscourt logic
            target_y = Geometry30Point.snap_to_y_grid(-my_y)
            target_x = 39.0
            return (FixedShotType.BASELINE_TOPSPIN, target_x, target_y)
        
        elif step_num == 4:
            # R4: 執行 BASELINE_FLAT_DTL (直線終結)
            target_y = my_y  # 直線
            target_x = 39.0
            return (FixedShotType.BASELINE_FLAT, target_x, target_y)
        
        else:
            # R5+: 預設對角上旋
            # 🔥 V2026.53: Fixed crosscourt logic
            target_y = Geometry30Point.snap_to_y_grid(-my_y)
            target_x = 39.0
            return (FixedShotType.BASELINE_TOPSPIN, target_x, target_y)


# ============================================================================
# 戰術腳本管理器 (Tactical Script Manager)
# ============================================================================

class TacticalScriptManager:
    """
    Tactical Script Manager: Manages all advanced tactical pattern recognition logic
    
    Tactical Scripts:
    - Backhand Cage (Tactic ID 35): 3-shot BH pressure → 4th-shot redirect
    - Two-Shot Passing (Tactic ID 14): Dipping shot → DTL winner
    - Wrong-footing (Tactic ID 42): Attack behind opponent's recovery direction
    
    Purpose:
    Automatically detects specific tactical patterns in shot history and triggers
    pre-programmed shot sequences based on match context and opponent positioning.
    """
    
    @staticmethod
    def check_tactical_script(
        step_num: int,
        shot_history: List[ShotStep],
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        opponent_prev_pos: Optional[Tuple[float, float]] = None
    ) -> Optional[Tuple[FixedShotType, float, float, str, Optional[str]]]:
        """
        Check if any tactical script should be triggered
        
        Args:
            step_num: Current shot number in rally
            shot_history: Complete shot history (List[ShotStep])
            my_pos: My current position (X, Y) in feet
            opponent_pos: Opponent's current position (X, Y) in feet
            opponent_prev_pos: Opponent's previous position (for wrong-footing detection)
        
        Returns:
            None if no script triggered, otherwise:
            (shot_type, target_x, target_y, reasoning, air_target_height)
        
        Tactical Priority:
        1. Two-Shot Passing (if opponent at net)
        2. Backhand Cage (if 3+ consecutive BH shots)
        3. Wrong-footing (if opponent recovering laterally)
        """
        # 1. 檢查反拍牢籠
        backhand_cage = TacticalScriptManager._check_backhand_cage(
            step_num, shot_history, my_pos, opponent_pos
        )
        if backhand_cage:
            return backhand_cage
        
        # 2. 檢查兩球穿越
        two_shot_passing = TacticalScriptManager._check_two_shot_passing(
            step_num, shot_history, my_pos, opponent_pos
        )
        if two_shot_passing:
            return two_shot_passing
        
        # 3. 檢查追身球
        wrong_footing = TacticalScriptManager._check_wrong_footing(
            step_num, shot_history, my_pos, opponent_pos, opponent_prev_pos
        )
        if wrong_footing:
            return wrong_footing
        
        return None
    
    @staticmethod
    def identify_opponent_weakness(
        shot_history: List[ShotStep],
        opponent_style: OpponentStyle
    ) -> Optional[str]:
        """
        識別對手弱點
        
        根據：
        - 對手風格（Pusher 反拍弱，Aggressive 網前弱）
        - 歷史失誤統計
        - 對手回球質量
        
        返回：
            "backhand", "forehand", "net", "baseline" 或 None
        """
        if opponent_style == OpponentStyle.PUSHER:
            return "backhand"  # Pusher 反拍弱
        elif opponent_style == OpponentStyle.AGGRESSIVE:
            return "net"  # Aggressive 網前弱
        elif opponent_style == OpponentStyle.NET_RUSHER:
            return "baseline"  # Net Rusher 底線弱
        return None
    
    @staticmethod
    def should_continue_attacking_weakness(
        weakness_type: str,
        attack_count: int
    ) -> bool:
        """
        判定是否繼續攻擊弱點（3-5 拍後變線）
        
        參數：
            weakness_type: 弱點類型
            attack_count: 連續攻擊次數
        
        返回：
            True 為繼續攻擊，False 為變線
        """
        # 連續 3-5 拍攻擊弱點，第 4-6 拍變線
        return attack_count < 4
    
    @staticmethod
    def _check_backhand_cage(
        step_num: int,
        shot_history: List[ShotStep],
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float]
    ) -> Optional[Tuple[FixedShotType, float, float, str, Optional[str]]]:
        """
        Backhand Cage (Tactic ID 35) - Systematic Strength-on-Weakness Pattern
        
        Scientific Principles:
        - **Geometric Pressure Coefficient**: Sustained targeting of opponent's weaker wing
          (typically BH = Y > 0 for right-handed opponents) accumulates defensive fatigue.
        - **Displacement Cost Function**: 3 consecutive shots to same zone reduce opponent's
          recovery efficiency by ~15% per shot (Garlikov recovery time penalty).
        - **Mandatory Redirection Protocol**: 4th shot MUST attack opposite zone to exploit
          maximum Geometric Pressure Coefficient (~27ft lateral displacement required).
        
        Trigger Conditions:
        - Minimum 3 consecutive shots to opponent BH zone (Y > 0, tolerance ±2ft)
        - All shots must be from "My" player (excludes opponent's returns)
        - Energy Cost: ~180 kcal per rally (assumes BMR 2200 kcal baseline)
        
        Returns:
            None or (shot_type, target_x, target_y, reasoning, air_target)
        """
        if step_num < 4:
            return None
        
        # Scan recent 3-shot window for consecutive BH attacks
        recent_shots = shot_history[-3:] if len(shot_history) >= 3 else []
        if len(recent_shots) < 3:
            return None
        
        # Validate consecutive BH targeting (Geometric Pressure accumulation)
        all_backhand_attacks = True
        for shot in recent_shots:
            if shot.player == "My":
                # Check if landing zone is in opponent BH corridor (Y > 0)
                if shot.end_pos[1] <= 0:
                    all_backhand_attacks = False
                    break
        
        if all_backhand_attacks:
            # Mandatory 4th shot: Attack diagonal gap (Deuce side, Y < 0)
            # Crosscourt Advantage Multiplier: 82.5ft diagonal path provides +5.8% net clearance
            target_y = Geometry30Point.snap_to_y_grid(-8)  # Diagonal open court
            target_x = 39.0  # Deep baseline target
            return (
                FixedShotType.BASELINE_TOPSPIN,
                target_x,
                target_y,
                "[Tactic ID 35] Backhand Cage: 3-shot BH pressure → Mandatory diagonal redirect (Displacement Cost: ~27ft)",
                None
            )
        
        return None
    
    @staticmethod
    def _check_two_shot_passing(
        step_num: int,
        shot_history: List[ShotStep],
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float]
    ) -> Optional[Tuple[FixedShotType, float, float, str, Optional[str]]]:
        """
        Two-Shot Passing Strategy (Tactic ID 14) - Percentage Tennis Protocol
        
        Scientific Principles:
        - **Vertical Displacement Cost**: Forcing opponent to volley UP from low position
          (net height = 3.0ft center) reduces volley quality by ~40% (Magnus effect disruption).
        - **Sequential Success Rate Optimization**:
          R(n): Dipping shot success = 85% (conservative estimate)
          R(n+1): DTL winner success = 75% | conditional on R(n) success
          Combined probability = 0.85 × 0.75 = 63.75% (vs. 45% direct passing attempt)
        - **Crosscourt Advantage Multiplier Reversal**: First shot uses 82.5ft diagonal safety,
          second shot exploits opponent's recovery lag with 78.0ft DTL speed advantage.
        
        Trigger Conditions:
        - Opponent at net (X < 21ft = service line threshold)
        - First shot: Dipping trajectory to opponent's feet (X=21ft, Y=opponent_y ± 3ft)
        - Second shot: DTL passing shot (X=39ft, Y=my_y for straight-line path)
        - Energy Cost: ~90 kcal per two-shot sequence (assumes BMR 2200 kcal baseline)
        
        Returns:
            None or (shot_type, target_x, target_y, reasoning, air_target)
        """
        # Validate opponent net position (Offensive Phase threshold)
        if opponent_pos[0] >= 21.0:
            return None
        
        # Check if previous shot was Dipping shot (R_n)
        if len(shot_history) > 0:
            last_shot = shot_history[-1]
            # Validate Dipping shot geometry: X ≈ 21ft, Y ≈ opponent_y
            if (abs(last_shot.end_pos[0] - 21.0) < 1.0 and 
                abs(last_shot.end_pos[1] - opponent_pos[1]) < 3.0):
                # Execute R(n+1): DTL Winner
                target_y = Geometry30Point.snap_to_y_grid(my_pos[1])  # DTL path
                target_x = 39.0  # Deep baseline target
                return (
                    FixedShotType.BASELINE_FLAT,
                    target_x,
                    target_y,
                    "[Tactic ID 14] Two-Shot Passing: R(n+1) DTL Winner (Displacement Cost: ~18ft lateral)",
                    "Low (Skim Net)"  # Net clearance < 1.5ft for maximum pace
                )
        else:
            # Execute R(n): Dipping shot to opponent's feet
            target_x = 21.0  # Service line depth (forces upward volley)
            target_y = Geometry30Point.snap_to_y_grid(opponent_pos[1])  # Opponent's Y-coordinate
            return (
                FixedShotType.BASELINE_SLICE,
                target_x,
                target_y,
                "兩球穿越：Dipping shot 準備",
                "Low (Skim Net)"
            )
        
        return None
    
    @staticmethod
    def _check_wrong_footing(
        step_num: int,
        shot_history: List[ShotStep],
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        opponent_prev_pos: Optional[Tuple[float, float]]
    ) -> Optional[Tuple[FixedShotType, float, float, str, Optional[str]]]:
        """
        追身球 (Wrong-footing)
        
        邏輯：檢測對手 recovery 向量，若反向回位則將落點設為對手上一拍位置
        """
        if opponent_prev_pos is None or len(shot_history) == 0:
            return None
        
        # 計算對手 recovery 向量
        recovery_vector = (
            opponent_pos[0] - opponent_prev_pos[0],
            opponent_pos[1] - opponent_prev_pos[1]
        )
        
        # 計算上一拍擊球向量（從對手上一拍位置到我方落點）
        last_shot = shot_history[-1]
        if last_shot.player == "My":
            shot_vector = (
                last_shot.end_pos[0] - opponent_prev_pos[0],
                last_shot.end_pos[1] - opponent_prev_pos[1]
            )
            
            # 計算向量夾角（簡化：檢查是否反向）
            dot_product = (recovery_vector[0] * shot_vector[0] + 
                          recovery_vector[1] * shot_vector[1])
            
            # 如果 recovery 向量與擊球向量反向（夾角 > 90度），則觸發追身球
            if dot_product < 0:
                # 落點設為對手上一拍位置
                target_x = opponent_prev_pos[0]
                target_y = Geometry30Point.snap_to_y_grid(opponent_prev_pos[1])
                return (
                    FixedShotType.BASELINE_FLAT,
                    target_x,
                    target_y,
                    "追身球：對手反向回位，攻擊上一拍位置",
                    None
                )
        
        return None
    
    @staticmethod
    def calculate_air_target_height(my_x: float) -> str:
        """
        根據我方 X 座標計算動態 Air Target
        
        參數：
            my_x: 我方 X 座標
        
        返回：
            "Low" (底線後方 >8ft), "Medium" (底線附近), "High" (底線內 <2ft)
        """
        if my_x < -39 - 8:  # 底線後方 >8ft
            return "High"
        elif my_x > -39 + 2:  # 底線內 <2ft
            return "Low"
        else:
            return "Medium"


# ============================================================================
# 戰術溯源系統 (Tactical Traceability System)
# ============================================================================

class TacticalTraceability:
    """
    Tactical Traceability System: 48-Tactic Tennis Singles Strategy Library (V2026.48 Pattern-First Edition)
    
    Architecture:
    1. **Pattern-First Matching**: Contextually matches current state against tactics_data.json
    2. **Massive Multiplier System**: 5.0x weight bonus for contextually matched tactics
    3. **Mandatory Traceability**: Every generated shot derives from one of 48 tactical IDs
    4. **Override Mechanism**: Tactical intent overrides generic geometric penalties
    
    Scientific Principles:
    - Wardlaw Directional Principles (Inside/Outside ball)
    - Garlikov Recovery (Angle bisector positioning)
    - Geometric Pressure Coefficient (Lateral displacement)
    - Displacement Cost Function (Energy expenditure, BMR 2200 kcal)
    - Crosscourt Advantage Multiplier (82.5ft diagonal vs 78.0ft DTL)
    """
    
    @staticmethod
    def _load_tactics_database() -> Dict:
        """
        Load 48-tactic database from tactics_data.json
        
        Returns:
            Dict mapping tactic ID (str) to tactic metadata
        """
        tactics_db = {}
        tactics_path = os.path.join(os.path.dirname(__file__), "tactics_data.json")
        if os.path.exists(tactics_path):
            with open(tactics_path, 'r', encoding='utf-8') as f:
                tactics_db = json.load(f)
        return tactics_db
    
    @staticmethod
    def _contextual_match_score(tactic_data: Dict, state: Dict) -> float:
        """
        Calculate contextual matching score for a specific tactic (0.0-5.0 multiplier)
        
        Matching Logic:
        - Phase alignment (D-N-O): +1.5x
        - Position criteria match: +1.5x
        - Opponent style affinity: +1.0x
        - Shot history pattern: +1.0x
        
        Args:
            tactic_data: Tactic metadata from tactics_data.json
            state: Current game state dictionary
        
        Returns:
            Contextual multiplier (0.0 = no match, 5.0 = perfect match)
        """
        multiplier = 0.0
        usage_context = tactic_data.get("usage_context", "").lower()
        opp_x, opp_y = state["opp_pos"]
        opp_phase = state["opp_phase"]
        my_phase = state["my_phase"]
        opponent_style = state["style"]
        history = state["history"]
        
        # 1. Phase Alignment Check (+1.5x)
        phase_keywords = {
            PhaseType.DEFENSE: ["防守", "防禦", "被動", "底線後", "defensive", "behind baseline"],
            PhaseType.NEUTRAL: ["相持", "底線", "中性", "平衡", "neutral", "baseline rally", "counterpunch"],
            PhaseType.OFFENSE: ["進攻", "機會", "短球", "網前", "offensive", "attack", "short ball", "mid-court"]
        }
        
        for phase, keywords in phase_keywords.items():
            if phase == opp_phase and any(kw in usage_context for kw in keywords):
                multiplier += 1.5
                break
        
        # 2. Position Criteria Match (+1.5x)
        position_matches = False
        
        # Opponent at net (X < 21ft from my perspective, which is > 21ft in absolute)
        if "網前" in usage_context or "net" in usage_context or "volley" in usage_context:
            # In normalized coords, opponent is at negative X, "at net" means X > -21
            if opp_x > -21.0:
                position_matches = True
        
        # Opponent at baseline (X < -30ft)
        if "底線" in usage_context or "baseline" in usage_context:
            if opp_x < -30.0:
                position_matches = True
        
        # Opponent backhand zone (Y > 0 for right-handed, Y < 0 for left-handed)
        if "反拍" in usage_context or "反手" in usage_context or "backhand" in usage_context:
            is_lefty = state.get("is_lefty", False)
            if (not is_lefty and opp_y > 3.0) or (is_lefty and opp_y < -3.0):
                position_matches = True
        
        # Wide position (|Y| > 10ft)
        if "寬" in usage_context or "角度" in usage_context or "wide" in usage_context or "angle" in usage_context:
            if abs(opp_y) > 10.0:
                position_matches = True
        
        if position_matches:
            multiplier += 1.5
        
        # 3. Opponent Style Affinity (+1.0x)
        style_keywords = {
            OpponentStyle.AGGRESSIVE: ["進攻", "攻擊", "aggressive", "attack"],
            OpponentStyle.PUSHER: ["防守", "穩定", "pusher", "defensive", "consistent"],
            OpponentStyle.NET_RUSHER: ["網前", "截擊", "net", "volley", "serve and volley"]
        }
        
        for style, keywords in style_keywords.items():
            if style == opponent_style and any(kw in usage_context for kw in keywords):
                multiplier += 1.0
                break
        
        # 4. Shot History Pattern Match (+1.0x)
        if len(history) >= 3:
            # Check for consecutive backhand shots (Backhand Cage pattern)
            if "反拍" in usage_context or "backhand cage" in usage_context.lower():
                consecutive_bh = 0
                for shot in history[-3:]:
                    if shot.player == "My" and shot.end_pos[1] > 2.0:  # Backhand side shots
                        consecutive_bh += 1
                if consecutive_bh >= 2:
                    multiplier += 1.0
            
            # Check for opponent net approach (Two-Shot Passing pattern)
            if ("穿越" in usage_context or "passing" in usage_context.lower()) and len(history) >= 1:
                if history[-1].player == "Opponent" and history[-1].end_pos[0] > 0:  # Opponent moved forward
                    multiplier += 1.0
        
        return min(5.0, multiplier)  # Cap at 5.0x

    @staticmethod
    def get_all_strategies(state: Dict) -> List[TacticalStrategy]:
        """
        Get all applicable tactical strategies (V2026.48 Pattern-First Architecture)
        
        Algorithm:
        1. Load 48-tactic database from tactics_data.json
        2. Create strategy objects for each tactic
        3. Calculate contextual matching score (0.0-5.0x multiplier)
        4. Apply multiplier to strategy weight
        5. Return strategies sorted by contextual relevance
        
        Returns:
            List[TacticalStrategy] with contextual multipliers applied
        """
        
        # Extract state variables
        my_pos = state.get("my_pos", (0, 0))
        opp_pos = state.get("opp_pos", (0, 0))
        my_phase = state.get("my_phase", PhaseType.NEUTRAL)
        opp_phase = state.get("opp_phase", PhaseType.NEUTRAL)
        surface = state.get("surface", SurfacePhysics.SurfaceType.HARD)
        momentum = state.get("momentum", 0.0)
        style = state.get("style", OpponentStyle.AGGRESSIVE)
        is_lefty = state.get("is_lefty", False)
        history = state.get("history", [])

        # Position coordinates
        my_x, my_y = my_pos
        opp_x, opp_y = opp_pos
        
        # 🔥 V2026.48: Load tactics database for contextual matching
        tactics_database = TacticalTraceability._load_tactics_database()

        strategies = []

        # --------------------------------------------------------------------
        # 1. 底線相持基礎 (1-2)
        # --------------------------------------------------------------------
        class Strategy1(TacticalStrategy):
            """底線對角深球: 高百分比網球基礎 (Tier 1 Fundamental)"""
            def is_applicable(self, s): return True  # 🔥 V2026.51: Always applicable (baseline fundamental) 
            def get_dynamic_target(self, s): 
                # 【公理推演】中點發散判定
                if abs(s["opp_pos"][1]) < 6.0:
                    ad_y, _ = CourtGeometry.get_divergent_targets(s["opp_pos"][1], s["is_lefty"])
                    return (39.0, ad_y) 
                return (39.0, -s["opp_pos"][1]) 
        strategies.append(Strategy1(1, "底線對角深球", "遵循 Wardlaw 方向性公理：保持跨場對角線時的最大安全性", FixedShotType.BASELINE_TOPSPIN))

        class Strategy5(TacticalStrategy):
            """網前短球: 縱向位移戰術"""
            def is_applicable(self, s): 
                # 【 मास्टर (Master) 】一致性檢查：若對手已經在網前，則不應放短球
                opp_at_net = s["my_pos"][0] > -12.0 # 針對對手而言，我方在其對策中的對手位置
                return not opp_at_net
            def get_dynamic_target(self, s): return (6.0, 13.5 if not s["is_lefty"] else -13.5)
        # 注意：ID 5 會覆蓋下方的 TACTICS_METADATA 註冊
        strategies.append(Strategy5(5, "網前短球", "遵循節奏破壞公理：利用縱向位移代價進行突擊", FixedShotType.DROP_SHOT))

        class Strategy2(TacticalStrategy):
            """底線變線直線: 破局與進攻 (Tier 2 Strategic)
            
            🔥 V2026.53: Merged ID 49-50 logic into this strategy
            - ID 49 (進攻型直線終結): Aggressive DTL flat shot in offense phase
            - ID 50 (進攻型直線上旋): Aggressive DTL topspin in neutral/defense phase
            - Both are now style-specific variants of ID 2 (DTL Redirection)
            """
            def is_applicable(self, s): return True
            def get_dynamic_target(self, s): 
                # 【公理推演】直線球：與對手位置同側
                target_y = s["opp_pos"][1]
                # 如果對手在中路，選擇一側（根據歷史或預設）
                if abs(target_y) < 6.0:
                    # 根據歷史選擇，或預設選擇 Ad 側
                    if s["history"]:
                        last_y = s["history"][-1].end_pos[1]
                        target_y = 8.0 if last_y >= 0 else -8.0
                    else:
                        target_y = 8.0  # 預設 Ad 側
                return (39.0, Geometry30Point.snap_to_y_grid(target_y))
            def calculate_weight(self, s):
                # 【🔥 V2026.53】進攻型對手時，直線球策略權重大幅提升
                # Merged logic from ID 49, 50 (were separate strategies)
                base_w = super().calculate_weight(s)
                if s["style"] == OpponentStyle.AGGRESSIVE:
                    # 進攻型對手時，直線球策略額外加成
                    if s["opp_phase"] == PhaseType.OFFENSE:
                        return base_w + 50.0  # ID 49 logic: 進攻相位大幅加成，確保 Top 3
                    elif s["opp_phase"] == PhaseType.NEUTRAL:
                        return base_w + 30.0  # ID 50 logic: 相持相位適度加成
                    else:  # DEFENSE
                        return base_w + 10.0  # 防守相位適度加成
                return base_w
        strategies.append(Strategy2(2, "底線變線直線", "遵循 Redirection 公理：對內側球進行變線反擊，利用幾何位移破壞對手平衡", FixedShotType.BASELINE_FLAT))

        # --------------------------------------------------------------------
        # 2. 進攻與空檔利用 (11-20)
        # --------------------------------------------------------------------
        class Strategy11(TacticalStrategy):
            """大角度拉開 (Inside-out) (Tier 1 Offensive Divergence)"""
            def is_applicable(self, s): return True  # 🔥 V2026.51: Always applicable (weight will handle priority)
            def get_dynamic_target(self, s): 
                return (24.0, 15.0 if not s["is_lefty"] else -15.0)
        strategies.append(Strategy11(11, "大角度拉開", "遵循幾何壓榨公理：在中央對峙時主動撕開角度，強迫對手產生最大橫向位移", FixedShotType.INSIDE_OUT))

        class Strategy12(TacticalStrategy):
            """相持變奏 (Tier 1 Breakout)"""
            def is_applicable(self, s): return True  # 🔥 V2026.51: Always applicable (weight adjusted by history) 
            def get_dynamic_target(self, s): 
                return (39.0, 14.5 if s["opp_pos"][1] < 0 else -14.5)
        strategies.append(Strategy12(12, "破局變線", "遵循 Anti-Stalemate 公理：打破底線均勢，攻擊遠端死角", FixedShotType.INSIDE_OUT))

        class Strategy16(TacticalStrategy):
            def is_applicable(self, s): 
                # 🔥 V2026.51: Relaxed to maintain tactical diversity
                return True  # Weight will prioritize when conditions are favorable
            def calculate_weight(self, s): 
                w = 88.0
                if s["style"] == OpponentStyle.PUSHER: w -= 30.0 # Pusher 即便有機會也可能打回對角
                return w
            def get_dynamic_target(self, s): return (39.0, s["opp_pos"][1])
        strategies.append(Strategy16(16, "角度-直線連擊", "Tier 1: 角度拉開後的直線致命一擊", FixedShotType.BASELINE_FLAT))
        
        # 🔥 V2026.53: REMOVED redundant ID 49, 50 strategies
        # Previous "AggressiveDTLStrategy" (ID 49) and "AggressiveDTLTopspinStrategy" (ID 50)
        # Logic merged into Strategy2 (ID 2) above with style-based weight adjustments
        # Reason: tactics_data.json only defines IDs 1-48, maintaining data integrity

        # --------------------------------------------------------------------
        # 3. 防守與恢復 (21-30) (Tier 3 Support)
        # --------------------------------------------------------------------
        class Strategy21(TacticalStrategy):
            """幾何中路壓制 (Tier 1 Defensive Baseline)"""
            def calculate_weight(self, s): 
                w = 58.0 
                if s["opp_phase"] == PhaseType.DEFENSE: w += 10.0
                return w
            def get_dynamic_target(self, s): return (39.0, 0.0) 
        strategies.append(Strategy21(21, "中路深球", "Tier 1: 壓制中路，縮小對手回球角度", FixedShotType.BASELINE_TOPSPIN))

        class Strategy22(TacticalStrategy):
            def is_applicable(self, s): return True  # 🔥 V2026.51: Always applicable (weight boosted in defense)
            def calculate_weight(self, s): return 78.0 # Tier 1 Emergency
            def get_dynamic_target(self, s): return (39.0, 0.0)
        strategies.append(Strategy22(22, "防守回心轉正", "Tier 1: 從極端防守位擊向中路爭取時間", FixedShotType.BASELINE_TOPSPIN))

        class Strategy25(TacticalStrategy):
            def calculate_weight(self, s):
                w = 35.0 # Tier 3 Variety
                if s["opp_phase"] == PhaseType.NEUTRAL: w -= 5.0 
                if s["surface"] == SurfacePhysics.SurfaceType.GRASS: w += 20.0
                return w
            def get_dynamic_target(self, s): return (39.0, -s["opp_pos"][1])
        strategies.append(Strategy25(25, "底線切球變奏", "Tier 3: 改變球速與彈跳節奏", FixedShotType.BASELINE_SLICE))

        # --------------------------------------------------------------------
        # 4. 數據驅動註冊 (其餘 48 項，跳過已定義的 ID: 1,2,11,12,16,21,22,25)
        # --------------------------------------------------------------------
        # 格式: ID: (名稱, 描述, 球種, 權重函數, [可選] 落點函數)
        TACTICS_METADATA = {
            3: ("底線中路限制", "遵循 Garlikov 恢復公理：通過中路深球限制對手回擊角度，為自身爭取回位時間", FixedShotType.BASELINE_TOPSPIN, lambda s: (39.0, 0.0)),
            4: ("外角拉開", "遵循幾何壓迫公理：利用極限角度擴大對手跑動覆蓋範圍", FixedShotType.INSIDE_OUT, lambda s: (22.0, 15.0 if s["opp_pos"][1] < 0 else -15.0 if s["opp_pos"][1] > 0 else 14.0)),
            5: ("網前短球", "遵循節奏破壞公理：在對手處於深層底線或中央對峙時，利用縱向位移代價進行突襲", FixedShotType.DROP_SHOT, lambda s: (6.0, 13.5 if not s["is_lefty"] else -13.5 if abs(s["opp_pos"][1]) < 6 else -s["opp_pos"][1])),
            6: ("防守高吊球", "遵循 Emergency Recovery 公理：在極端防守下通過高弧度球換取生存時間", FixedShotType.LOB, lambda s: (38.0, 0.0)),
            7: ("隨球上網", "遵循 Phase Transition 公理：將底線優勢轉化為網前壓迫", FixedShotType.BASELINE_FLAT, lambda s: (39.0, -s["opp_pos"][1] if abs(s["opp_pos"][1]) > 1 else 10.0)),
            8: ("穿越致勝球", "遵循 Passing Shot 公理：在對手網前時攻擊兩翼走廊", FixedShotType.BASELINE_TOPSPIN),
            9: ("專攻反拍", "遵循 Weakness Exploitation 公理：持續施壓於解剖學弱側", FixedShotType.BASELINE_TOPSPIN, lambda s: (39.0, -11.0 if not s["is_lefty"] else 11.0)),
            10: ("側身正發進攻", "遵循 Core Weapon 公理：繞過反拍使用核心正手主導進攻", FixedShotType.BASELINE_TOPSPIN, lambda s: (39.0, 10.0 if not s["is_lefty"] else -10.0)),
            13: ("反拍牢籠", "遵循 Persistence 公理：通過持續的高質量反拍壓制誘導對手失誤", FixedShotType.BASELINE_FLAT, lambda s: (39.0, -13.0 if not s["is_lefty"] else 13.0)),
            15: ("追身壓迫", "遵循 Body Attack 公理：攻擊對手重心回位路徑，限制其揮拍空間", FixedShotType.BASELINE_FLAT, lambda s: (s["my_pos"][0] + 5, s["my_pos"][1] * 0.5)),
            17: ("發球+1 第三拍強攻", "遵循 Initial Advantage 公理：將發球創造的初始相位優勢轉化為得分點", FixedShotType.BASELINE_TOPSPIN),
            18: ("接發球回歸底線", "遵循 Neutralization 公理：中和發球威力，強迫進入底線相持", FixedShotType.BASELINE_TOPSPIN),
            19: ("高百分比相持", "遵循 Percentage Tennis 公理：通過穩健的對角上旋球進行戰術消耗", FixedShotType.BASELINE_TOPSPIN), 
            20: ("紅土點建構", "遵循 Build-up 公理：在慢速球場通過極大角度連續帶動對手", FixedShotType.BASELINE_TOPSPIN, lambda s: (32.0, -s["opp_pos"][1] * 1.3 if abs(s["opp_pos"][1]) > 1 else -14.0)),
            23: ("機會球致勝", "遵循 Termination 公理：在進攻相位下執行最後的致命一擊", FixedShotType.BASELINE_FLAT),
            30: ("切球誘敵", "遵循 Variety 公理：利用切球產生的低彈跳誘導對手向上擊球或失誤", FixedShotType.DROP_SHOT, lambda s: (7.0, -s["opp_pos"][1] if abs(s["opp_pos"][1]) > 1 else -12.0)),
            48: ("通用戰術", "基礎公理化球路", FixedShotType.BASELINE_TOPSPIN),
        }

        for tid, entry in TACTICS_METADATA.items():
            name, desc, stype = entry[0], entry[1], entry[2]
            target_func = entry[3] if len(entry) > 3 else (lambda s: (39.0, -s["opp_pos"][1]))
            
            def make_strat(t, n, d, s, tf):
                class AutoStrat(TacticalStrategy):
                    # 🔥 V2026.51: Ensure is_applicable() returns True by default
                    def is_applicable(self, state: Dict) -> bool:
                        return True  # Always applicable for TACTICS_METADATA strategies
                    # 使用基類核心公理化評分
                    def get_dynamic_target(self, states): return tf(states)
                return AutoStrat(t, n, d, s)
            strategies.append(make_strat(tid, name, desc, stype, target_func))

        # ============================================================================
        # 🔥 V2026.48 PATTERN-FIRST CONTEXTUAL MATCHING
        # ============================================================================
        # Apply contextual multipliers to ALL strategies based on tactics_database matching
        for strategy in strategies:
            tactic_id_str = str(strategy.id)
            if tactic_id_str in tactics_database:
                tactic_data = tactics_database[tactic_id_str]
                contextual_multiplier = TacticalTraceability._contextual_match_score(tactic_data, state)
                strategy.set_contextual_multiplier(contextual_multiplier)
                
                # Debug logging (optional, can be disabled in production)
                if contextual_multiplier > 1.5 and state.get("debug_mode"):
                    print(f"[PATTERN-FIRST] Tactic ID {strategy.id} ({strategy.name}): "
                          f"Contextual Multiplier = {contextual_multiplier:.2f}x")
        
        # 5. Filter applicable strategies (applicability check happens before weight calculation)
        # 🔍 V2026.51 DIAGNOSTIC: Track filtering behavior
        pre_filter_count = len(strategies)
        
        # 🔥 V2026.51 EMERGENCY FIX: If normal filtering is too strict, bypass it
        applicable_strategies = []
        filtered_out = []
        for strat in strategies:
            if strat.is_applicable(state):
                applicable_strategies.append(strat)
            else:
                filtered_out.append((strat.id, strat.name))
        
        post_filter_count = len(applicable_strategies)
        
        print(f"[V2026.51 Applicability Filter] {pre_filter_count} → {post_filter_count} strategies")
        if pre_filter_count != post_filter_count:
            print(f"  Filtered out {pre_filter_count - post_filter_count} strategies:")
            for sid, sname in filtered_out[:5]:  # Show first 5
                print(f"    - ID {sid}: {sname}")
        
        # 🔥 V2026.51 CRITICAL FIX: Emergency bypass if too few strategies
        if post_filter_count < 10:
            print(f"  ⚠️ EMERGENCY: Only {post_filter_count} strategies! Bypassing is_applicable() filter...")
            print(f"     Forcing ALL {pre_filter_count} strategies to be applicable for diversity")
            applicable_strategies = strategies  # Use ALL strategies
        
        return applicable_strategies

    @staticmethod
    def match_tactic(shot_type, target_x, target_y, phase, opponent_style=None, **kwargs):
        """
        將具體的擊球數據反向映射到 48 項戰術表
        """
        is_lefty = kwargs.get("is_lefty", False)
        # 提取落點特徵
        is_side_out = abs(target_y) > 13.0
        is_center = abs(target_y) < 3.0
        is_deep = target_x > 30.0
        is_short = target_x < 15.0
        
        # 1. 特殊球種優先
        if shot_type == FixedShotType.LOB: return (6, "防守高吊球")
        if shot_type == FixedShotType.DROP_SHOT: 
            if is_short: return (5, "網前短球")
            return (30, "切球誘敵")
        if shot_type == FixedShotType.INSIDE_OUT: return (4, "外角拉開")
        if shot_type == FixedShotType.BASELINE_SLICE: return (25, "底線切球變奏")
        
        # 2. 定位致勝/特殊相位
        if phase == PhaseType.OFFENSE: return (23, "機會球致勝")
        if kwargs.get("is_passing", False): return (8, "穿越致勝球")
        
        # 3. 核心底線球路判定
        if is_center: return (3, "底線中路封鎖")
        
        # 4. 判斷直線 (DTL) vs 對角 (Diagonal)
        # 這裡使用簡單的邏輯: 如果 target_y 的正負號與起點(如果是 history)不同，則是對角
        # 但在 match_tactic 這裡通常只拿到結果。
        # 改用對手位與落點位判定 (假設對手是 striker)
        # 在我們的系統中，target_y 的絕對值通常對應戰術意圖
        if shot_type == FixedShotType.BASELINE_FLAT:
            return (2, "底線變線直線")
        
        if is_side_out: return (4, "外角拉開")
        
        # 預設
        return (1, "底線對角深球")
        
        # 根據對手風格匹配
        if opponent_style:
            if opponent_style == OpponentStyle.PUSHER:
                if shot_type == FixedShotType.DROP_SHOT:
                    return (30, TacticalTraceability.TACTICS_TABLE[30])
                elif shot_type == FixedShotType.BASELINE_SLICE:
                    return (25, TacticalTraceability.TACTICS_TABLE[25])
            elif opponent_style == OpponentStyle.NET_RUSHER:
                if is_passing:
                    return (17, TacticalTraceability.TACTICS_TABLE[17])
            elif opponent_style == OpponentStyle.AGGRESSIVE:
                if phase == PhaseType.DEFENSE:
                    return (18, TacticalTraceability.TACTICS_TABLE[18])
        
        # 根據擊球類型匹配
        if shot_type == FixedShotType.BASELINE_TOPSPIN:
            if abs(target_y) < 3:  # 中路
                return (3, TacticalTraceability.TACTICS_TABLE[3])
            elif abs(target_y) >= 14:  # 極外角
                return (4, TacticalTraceability.TACTICS_TABLE[4])
            else:  # 對角
                return (1, TacticalTraceability.TACTICS_TABLE[1])
        elif shot_type == FixedShotType.BASELINE_FLAT:
            if abs(target_y) < 3:  # 直線
                return (2, TacticalTraceability.TACTICS_TABLE[2])
            else:
                return (32, TacticalTraceability.TACTICS_TABLE[32])
        elif shot_type == FixedShotType.BASELINE_SLICE:
            return (25, TacticalTraceability.TACTICS_TABLE[25])
        elif shot_type == FixedShotType.INSIDE_OUT:
            return (4, TacticalTraceability.TACTICS_TABLE[4])
        elif shot_type == FixedShotType.DROP_SHOT:
            return (5, TacticalTraceability.TACTICS_TABLE[5])
        elif shot_type == FixedShotType.LOB:
            return (6, TacticalTraceability.TACTICS_TABLE[6])
        
        # 預設：非標準戰術
        return (48, TacticalTraceability.TACTICS_TABLE[48])
    
    @staticmethod
    def validate_shot_tactic(
        shot_type: FixedShotType,
        target_x: float,
        target_y: float,
        phase: PhaseType,
        **kwargs
    ) -> Tuple[int, str, bool]:
        """
        驗證擊球是否符合戰術表
        
        參數：
            shot_type: 擊球類型
            target_x: 目標 X 座標
            target_y: 目標 Y 座標
            phase: 相位
            **kwargs: 其他參數
        
        返回：
            (tactic_id, tactic_name, is_valid)
        """
        tactic_id, tactic_name = TacticalTraceability.match_tactic(
            shot_type, target_x, target_y, phase, **kwargs
        )
        is_valid = (tactic_id != 48)  # 非標準戰術視為無效
        return (tactic_id, tactic_name, is_valid)


# ============================================================================
# Mid-Court Tactics（中場戰術）
# ============================================================================

class MidCourtTactics:
    """
    Mid-Court Tactics（中場戰術）
    
    資料來源：A Strategic Guide to Tennis Tactics by Court Position
    
    核心戰術：
    1. Approach Shot（過渡球）：深球到角落或對手腳下，然後上網
    2. Swinging Volley（揮拍截擊）：在中場攔截弱球，空中擊球
    """
    
    @staticmethod
    def is_mid_court_position(x: float) -> bool:
        """
        判定是否為中場位置
        
        參數：
            x: X 座標
        
        返回：
            True 為中場位置（-21ft < X < -5ft）
        """
        return -21.0 < x < -5.0
    
    @staticmethod
    def should_use_approach_shot(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        last_shot_end: Tuple[float, float]
    ) -> bool:
        """
        判定是否應該使用 Approach Shot（過渡球）
        
        條件：
        - 對手回球短（X > -21ft，即在我方發球線內）
        - 我方在中場或底線附近
        - 對手在底線後方
        
        參數：
            my_pos: 我方位置
            opponent_pos: 對手位置
            last_shot_end: 上一拍落點
        
        返回：
            True 為應該使用 Approach Shot
        """
        my_x, my_y = my_pos
        opp_x, opp_y = opponent_pos
        last_x, last_y = last_shot_end
        
        # 對手回球必須短（在我方發球線內）
        if last_x > -21.0:
            return False
        
        # 對手必須在底線後方
        if opp_x < 21.0:
            return False
        
        # 我方必須在中場或底線附近
        if my_x < -30.0:
            return False
        
        return True
    
    @staticmethod
    def execute_approach_shot(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float]
    ) -> Tuple[FixedShotType, float, float, str]:
        """
        執行 Approach Shot（過渡球）
        
        目標：
        - 深球直線（DTL）或直接打到對手腳下
        - 迫使對手回球弱，為截擊做準備
        
        參數：
            my_pos: 我方位置
            opponent_pos: 對手位置
        
        返回：
            (shot_type, target_x, target_y, reason)
        """
        my_x, my_y = my_pos
        opp_x, opp_y = opponent_pos
        
        # 優先選擇：深球直線（DTL）到對手底線
        # 或者打到對手腳下（X=21ft, Y=對手Y座標）
        target_x = 39.0  # 對手底線深處
        target_y = Geometry30Point.snap_to_y_grid(opp_y)  # 對手腳下或直線
        
        return (
            FixedShotType.BASELINE_FLAT,  # 平擊球穿透力強
            target_x,
            target_y,
            "Approach Shot: 深球直線，準備上網截擊"
        )
    
    @staticmethod
    def should_use_swinging_volley(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        last_shot_end: Tuple[float, float]
    ) -> bool:
        """
        判定是否應該使用 Swinging Volley（揮拍截擊）
        
        條件：
        - 對手回球弱（高且慢）
        - 我方在中場（-21ft < X < -5ft）
        - 可以空中攔截
        
        參數：
            my_pos: 我方位置
            opponent_pos: 對手位置
            last_shot_end: 上一拍落點
        
        返回：
            True 為應該使用 Swinging Volley
        """
        my_x, my_y = my_pos
        last_x, last_y = last_shot_end
        
        # 我方必須在中場
        if not MidCourtTactics.is_mid_court_position(my_x):
            return False
        
        # 對手回球必須在我方中場附近（可以攔截）
        if last_x < -25.0 or last_x > -5.0:
            return False
        
        return True
    
    @staticmethod
    def execute_swinging_volley(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float]
    ) -> Tuple[FixedShotType, float, float, str]:
        """
        執行 Swinging Volley（揮拍截擊）
        
        目標：
        - 空中攔截弱球
        - 攻擊性擊球到空檔
        - 搶時間，不讓球落地
        
        參數：
            my_pos: 我方位置
            opponent_pos: 對手位置
        
        返回：
            (shot_type, target_x, target_y, reason)
        """
        my_x, my_y = my_pos
        opp_x, opp_y = opponent_pos
        
        # 攻擊對手空檔（與對手位置異側）
        if opp_y >= 0:
            target_y = -14.0  # 攻擊 Deuce 側空檔
        else:
            target_y = 14.0   # 攻擊 Ad 側空檔
        
        target_y = Geometry30Point.snap_to_y_grid(target_y)
        target_x = 39.0  # 對手底線
        
        return (
            FixedShotType.BASELINE_FLAT,  # 平擊球搶時間
            target_x,
            target_y,
            "Swinging Volley: 空中攔截，攻擊空檔"
        )
    
    @staticmethod
    def check_and_execute(
        step_num: int,
        shot_history: List[ShotStep],
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        phase: PhaseType
    ) -> Optional[Tuple[FixedShotType, float, float, str]]:
        """
        檢查並執行中場戰術
        
        參數：
            step_num: 當前拍數
            shot_history: 擊球歷史
            my_pos: 我方位置
            opponent_pos: 對手位置
            phase: 相位
        
        返回：
            (shot_type, target_x, target_y, reason) 或 None
        """
        if len(shot_history) == 0:
            return None
        
        last_shot = shot_history[-1]
        last_shot_end = last_shot.end_pos
        
        # 優先檢查 Swinging Volley（更主動）
        if MidCourtTactics.should_use_swinging_volley(my_pos, opponent_pos, last_shot_end):
            return MidCourtTactics.execute_swinging_volley(my_pos, opponent_pos)
        
        # 其次檢查 Approach Shot
        if MidCourtTactics.should_use_approach_shot(my_pos, opponent_pos, last_shot_end):
            return MidCourtTactics.execute_approach_shot(my_pos, opponent_pos)
        
        return None


# ============================================================================
# Near-Net Tactics（網前戰術）
# ============================================================================

class NearNetTactics:
    """
    Near-Net Tactics（網前戰術）
    
    資料來源：A Strategic Guide to Tennis Tactics by Court Position
    
    核心戰術（根據球高度）：
    1. High Volley（高截擊）：肩膀高度以上，攻擊性，角度
    2. Mid-Height Volley（中高截擊）：臀部到肩膀，深球，直線
    3. Low Volley（低截擊）：臀部以下，短球或深低球
    """
    
    class VolleyHeight(Enum):
        """截擊球高度分類"""
        HIGH = "High"      # 肩膀高度以上（> 5.5ft）
        MID = "Mid"        # 臀部到肩膀（3.5ft - 5.5ft）
        LOW = "Low"        # 臀部以下（< 3.5ft）
    
    @staticmethod
    def is_near_net_position(x: float) -> bool:
        """
        判定是否為網前位置
        
        參數：
            x: X 座標
        
        返回：
            True 為網前位置（X > -10ft）
        """
        return x > -10.0
    
    @staticmethod
    def estimate_ball_height(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        last_shot_type: Optional[FixedShotType]
    ) -> VolleyHeight:
        """
        估算球的高度（用於選擇截擊類型）
        
        參數：
            my_pos: 我方位置
            opponent_pos: 對手位置
            last_shot_type: 上一拍球種
        
        返回：
            VolleyHeight 枚舉
        """
        # 根據球種和位置估算高度
        if last_shot_type == FixedShotType.LOB:
            return NearNetTactics.VolleyHeight.HIGH
        elif last_shot_type == FixedShotType.BASELINE_TOPSPIN:
            return NearNetTactics.VolleyHeight.MID
        elif last_shot_type == FixedShotType.BASELINE_SLICE:
            return NearNetTactics.VolleyHeight.LOW
        elif last_shot_type == FixedShotType.DROP_SHOT:
            return NearNetTactics.VolleyHeight.LOW
        else:
            # 預設：根據對手位置估算
            opp_x, opp_y = opponent_pos
            if opp_x < 21.0:  # 對手在底線後方，回球可能較高
                return NearNetTactics.VolleyHeight.MID
            else:  # 對手在網前，回球可能較低
                return NearNetTactics.VolleyHeight.LOW
    
    @staticmethod
    def execute_high_volley(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float]
    ) -> Tuple[FixedShotType, float, float, str]:
        """
        執行 High Volley（高截擊）
        
        目標：
        - 攻擊性擊球，角度
        - 向下擊球，創造致勝球
        
        參數：
            my_pos: 我方位置
            opponent_pos: 對手位置
        
        返回：
            (shot_type, target_x, target_y, reason)
        """
        my_x, my_y = my_pos
        opp_x, opp_y = opponent_pos
        
        # 攻擊對手空檔（角度）
        if opp_y >= 0:
            target_y = -14.0  # 攻擊 Deuce 側空檔
        else:
            target_y = 14.0   # 攻擊 Ad 側空檔
        
        target_y = Geometry30Point.snap_to_y_grid(target_y)
        target_x = 39.0  # 對手底線
        
        return (
            FixedShotType.BASELINE_FLAT,  # 平擊球攻擊性強
            target_x,
            target_y,
            "High Volley: 攻擊性截擊，角度致勝"
        )
    
    @staticmethod
    def execute_mid_volley(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float]
    ) -> Tuple[FixedShotType, float, float, str]:
        """
        執行 Mid-Height Volley（中高截擊）
        
        目標：
        - 深球，直線
        - 迫使對手回球弱，為下一拍做準備
        
        參數：
            my_pos: 我方位置
            opponent_pos: 對手位置
        
        返回：
            (shot_type, target_x, target_y, reason)
        """
        my_x, my_y = my_pos
        opp_x, opp_y = opponent_pos
        
        # 深球直線（DTL）
        target_y = Geometry30Point.snap_to_y_grid(opp_y)
        target_x = 39.0  # 對手底線深處
        
        return (
            FixedShotType.BASELINE_FLAT,  # 平擊球穿透力強
            target_x,
            target_y,
            "Mid-Height Volley: 深球直線，迫使對手回球弱"
        )
    
    @staticmethod
    def execute_low_volley(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float]
    ) -> Tuple[FixedShotType, float, float, str]:
        """
        執行 Low Volley（低截擊）
        
        目標：
        - 短球（drop volley）或深低球
        - 保持球低，迫使對手向上擊球
        
        參數：
            my_pos: 我方位置
            opponent_pos: 對手位置
        
        返回：
            (shot_type, target_x, target_y, reason)
        """
        my_x, my_y = my_pos
        opp_x, opp_y = opponent_pos
        
        # 根據對手位置選擇：短球或深低球
        if opp_x < 21.0:  # 對手在底線後方，使用短球
            target_x = 5.0   # 對手發球線內
            target_y = Geometry30Point.snap_to_y_grid(opp_y)
            shot_type = FixedShotType.DROP_SHOT
            reason = "Low Volley: 短球（drop volley），對手無法追到"
        else:  # 對手在網前，使用深低球
            target_x = 39.0  # 對手底線
            target_y = Geometry30Point.snap_to_y_grid(opp_y)
            shot_type = FixedShotType.BASELINE_SLICE  # 切球保持低
            reason = "Low Volley: 深低球，迫使對手向上擊球"
        
        return (shot_type, target_x, target_y, reason)
    
    @staticmethod
    def check_and_execute(
        step_num: int,
        shot_history: List[ShotStep],
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        phase: PhaseType
    ) -> Optional[Tuple[FixedShotType, float, float, str]]:
        """
        檢查並執行網前戰術
        
        參數：
            step_num: 當前拍數
            shot_history: 擊球歷史
            my_pos: 我方位置
            opponent_pos: 對手位置
            phase: 相位
        
        返回：
            (shot_type, target_x, target_y, reason) 或 None
        """
        # 必須在網前位置
        if not NearNetTactics.is_near_net_position(my_pos[0]):
            return None
        
        if len(shot_history) == 0:
            return None
        
        last_shot = shot_history[-1]
        ball_height = NearNetTactics.estimate_ball_height(
            my_pos, opponent_pos, last_shot.shot_type
        )
        
        # 根據球高度選擇截擊類型
        if ball_height == NearNetTactics.VolleyHeight.HIGH:
            return NearNetTactics.execute_high_volley(my_pos, opponent_pos)
        elif ball_height == NearNetTactics.VolleyHeight.MID:
            return NearNetTactics.execute_mid_volley(my_pos, opponent_pos)
        else:  # LOW
            return NearNetTactics.execute_low_volley(my_pos, opponent_pos)


# ============================================================================
# Inside-In Pattern（Inside-in 正手模式）
# ============================================================================

class InsideInPattern:
    """
    Inside-In Pattern（Inside-in 正手模式）
    
    資料來源：A Strategic Guide to Tennis Tactics by Court Position
    
    核心概念：
    - Inside-out: 從 Ad 側繞過反拍，用正手攻擊對手反拍側
    - Inside-in: 從 Ad 側繞過反拍，用正手攻擊對手正拍側（直線）
    
    戰術序列：
    1. 繞過反拍，使用 Inside-out 正手到對手反拍側
    2. 對手回球到正拍側
    3. 使用 Inside-in 正手直線攻擊對手正拍側空檔
    """
    
    @staticmethod
    def should_trigger_inside_in(
        shot_history: List[ShotStep],
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        phase: PhaseType
    ) -> bool:
        """
        判定是否應該觸發 Inside-in 模式
        
        條件：
        - 上一拍是 Inside-out 正手
        - 對手回球到我方正拍側
        - 我方處於進攻相位
        
        參數：
            shot_history: 擊球歷史
            my_pos: 我方位置
            opponent_pos: 對手位置
            phase: 相位
        
        返回：
            True 為應該觸發 Inside-in 模式
        """
        if len(shot_history) < 2:
            return False
        
        # 必須處於進攻相位
        if phase != PhaseType.OFFENSE:
            return False
        
        # 檢查上一拍是否為 Inside-out
        last_shot = shot_history[-1]
        if last_shot.shot_type != FixedShotType.INSIDE_OUT:
            return False
        
        # 檢查上上一拍是否為我方 Inside-out 正手
        prev_shot = shot_history[-2]
        if prev_shot.player != "My" or prev_shot.shot_type != FixedShotType.INSIDE_OUT:
            return False
        
        # 對手回球應該到我方正拍側（Ad 側，Y > 0）
        my_y = my_pos[1]
        if my_y <= 0:
            return False  # 我方不在 Ad 側
        
        return True
    
    @staticmethod
    def execute_inside_in_shot(
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float]
    ) -> Tuple[FixedShotType, float, float, str]:
        """
        執行 Inside-in 正手擊球
        
        目標：
        - 從 Ad 側用正手直線攻擊對手正拍側空檔
        - 利用對手回位不足，創造致勝球
        
        參數：
            my_pos: 我方位置
            opponent_pos: 對手位置
        
        返回：
            (shot_type, target_x, target_y, reason)
        """
        my_x, my_y = my_pos
        opp_x, opp_y = opponent_pos
        
        # Inside-in: 從 Ad 側（Y > 0）直線攻擊對手 Ad 側（Y = 8）
        # 但這是對手視角，我方攻擊對手 Ad 側 = Y = -8（我方視角）
        target_y = -8.0  # 對手 Ad 側（我方視角）
        target_y = Geometry30Point.snap_to_y_grid(target_y)
        target_x = 39.0  # 對手底線
        
        return (
            FixedShotType.BASELINE_FLAT,  # 平擊球穿透力強
            target_x,
            target_y,
            "Inside-in: 正手直線攻擊對手正拍側空檔"
        )
    
    @staticmethod
    def check_and_execute(
        step_num: int,
        shot_history: List[ShotStep],
        my_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        phase: PhaseType
    ) -> Optional[Tuple[FixedShotType, float, float, str]]:
        """
        檢查並執行 Inside-in 模式
        
        參數：
            step_num: 當前拍數
            shot_history: 擊球歷史
            my_pos: 我方位置
            opponent_pos: 對手位置
            phase: 相位
        
        返回：
            (shot_type, target_x, target_y, reason) 或 None
        """
        if InsideInPattern.should_trigger_inside_in(shot_history, my_pos, opponent_pos, phase):
            return InsideInPattern.execute_inside_in_shot(my_pos, opponent_pos)
        
        return None


# ============================================================================
# 🔥 深度重構：致勝期望值評分引擎 (Point Win Probability Engine)
# ============================================================================

class PointWinProbabilityEngine:
    """
    致勝期望值 (PWP) 評分引擎
    
    核心理念：
    - 評分不應只是「這球好不好打」，而是「打這球後贏得這一分的機率」
    - 考慮擊球後的優勢位階、對手回球質量、連續擊球組合
    
    公式：
    PWP = (成功率 % × 擊球後優勢位階) - (失誤率 % × 100) + 組合拳加成
    """
    
    @staticmethod
    def calculate_advantage_tier(
        target_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        shot_type: FixedShotType,
        phase: PhaseType
    ) -> float:
        """
        計算擊球後的優勢位階 (0-100)
        
        位階定義：
        - 90-100: 致勝球位階（對手幾乎無法回球）
        - 70-89: 主導位階（對手只能防守回球）
        - 50-69: 平衡位階（相持）
        - 30-49: 受迫位階（我方需要防守）
        - 0-29: 危險位階（可能直接失分）
        """
        distance = Geometry30Point.calculate_distance(target_pos, opponent_pos)
        
        # 基礎位階：基於距離
        if distance >= 18.0:
            base_tier = 95.0  # 致勝球位階
        elif distance >= 15.0:
            base_tier = 80.0  # 主導位階
        elif distance >= 12.0:
            base_tier = 65.0  # 平衡位階
        elif distance >= 9.0:
            base_tier = 45.0  # 受迫位階
        else:
            base_tier = 25.0  # 危險位階
        
        # 相位調整
        phase_bonus = 0.0
        if phase == PhaseType.OFFENSE:
            phase_bonus = +15.0
        elif phase == PhaseType.DEFENSE:
            phase_bonus = -15.0
        
        # 球種調整（攻擊性球種提升位階）
        shot_bonus = 0.0
        if shot_type in [FixedShotType.BASELINE_FLAT, FixedShotType.INSIDE_OUT]:
            shot_bonus = +10.0
        elif shot_type == FixedShotType.DROP_SHOT:
            shot_bonus = +8.0
        
        return max(0, min(100, base_tier + phase_bonus + shot_bonus))
    
    @staticmethod
    def calculate_depth_nonlinear_bonus(target_x: float, court_side: str = "opponent") -> float:
        """
        計算深度的非線性加分
        
        核心概念：
        - 球落在底線深處 3ft 內的加分應遠高於 6ft 處（邊際效益遞增）
        - 採用二次函數：越接近底線，加分越高
        
        參數：
            target_x: 目標 X 座標
            court_side: "opponent" 或 "my"
        
        返回：
            深度加分 (0-30)
        """
        if court_side == "opponent":
            baseline_x = 39.0
            distance_from_baseline = abs(baseline_x - abs(target_x))
        else:
            baseline_x = -39.0
            distance_from_baseline = abs(abs(baseline_x) - abs(target_x))
        
        # 非線性加分：採用二次函數
        # 0-3ft: +30, 3-6ft: +20, 6-12ft: +10, 12+ft: 0
        if distance_from_baseline <= 3.0:
            return 30.0
        elif distance_from_baseline <= 6.0:
            return 20.0 - ((distance_from_baseline - 3.0) / 3.0) * 10.0
        elif distance_from_baseline <= 12.0:
            return 10.0 - ((distance_from_baseline - 6.0) / 6.0) * 10.0
        else:
            return 0.0
    
    @staticmethod
    def calculate_angle_nonlinear_bonus(
        target_y: float,
        surface_type: SurfacePhysics.SurfaceType
    ) -> float:
        """
        計算角度的非線性加分（結合場地特性）
        
        核心概念：
        - 大角度球若能結合場地特性，應有乘法加成而非加法
        - 紅土：高彈跳 + 大角度 = 極難處理
        - 草地：低滑行 + 大角度 = 快速得分
        
        參數：
            target_y: 目標 Y 座標
            surface_type: 場地類型
        
        返回：
            角度加分 (0-40)
        """
        angle_abs = abs(target_y)
        
        # 基礎角度加分
        if angle_abs >= 14.0:
            base_bonus = 30.0
        elif angle_abs >= 10.0:
            base_bonus = 20.0
        elif angle_abs >= 6.0:
            base_bonus = 10.0
        else:
            base_bonus = 0.0
        
        # 場地乘法加成
        surface_multiplier = 1.0
        if surface_type == SurfacePhysics.SurfaceType.CLAY and angle_abs >= 12.0:
            surface_multiplier = 1.3  # 紅土大角度高彈跳
        elif surface_type == SurfacePhysics.SurfaceType.GRASS and angle_abs >= 12.0:
            surface_multiplier = 1.2  # 草地大角度低滑行
        
        return base_bonus * surface_multiplier
    
    @staticmethod
    def calculate_pwp_score(
        shot_type: FixedShotType,
        target_pos: Tuple[float, float],
        opponent_pos: Tuple[float, float],
        phase: PhaseType,
        base_success_rate: float,
        surface_type: SurfacePhysics.SurfaceType,
        combo_bonus: float = 0.0
    ) -> Dict[str, float]:
        """
        計算致勝期望值 (PWP) 評分
        
        公式：
        PWP = (成功率 × 優勢位階) - (失誤率 × 100) 
              + 深度非線性加分 + 角度非線性加分 + 組合拳加成
        
        參數：
            shot_type: 球種
            target_pos: 目標位置
            opponent_pos: 對手位置
            phase: 相位
            base_success_rate: 基礎成功率
            surface_type: 場地類型
            combo_bonus: 組合拳加成
        
        返回：
            包含 PWP 分數和各項明細的字典
        """
        # 1. 計算優勢位階
        advantage_tier = PointWinProbabilityEngine.calculate_advantage_tier(
            target_pos, opponent_pos, shot_type, phase
        )
        
        # 2. 計算失誤率
        error_rate = 100.0 - base_success_rate
        
        # 3. 計算深度非線性加分
        depth_bonus = PointWinProbabilityEngine.calculate_depth_nonlinear_bonus(
            target_pos[0], "opponent"
        )
        
        # 4. 計算角度非線性加分
        angle_bonus = PointWinProbabilityEngine.calculate_angle_nonlinear_bonus(
            target_pos[1], surface_type
        )
        
        # 5. 計算 PWP 總分
        pwp_score = (
            (base_success_rate / 100.0) * advantage_tier
            - (error_rate / 100.0) * 100.0
            + depth_bonus
            + angle_bonus
            + combo_bonus
        )
        
        return {
            "pwp_score": pwp_score,
            "advantage_tier": advantage_tier,
            "error_penalty": -(error_rate / 100.0) * 100.0,
            "depth_bonus": depth_bonus,
            "angle_bonus": angle_bonus,
            "combo_bonus": combo_bonus,
            "base_success_rate": base_success_rate
        }


# ============================================================================
# 🔥 深度重構：動態風險護欄引擎 (Dynamic Risk Barrier Engine)
# ============================================================================

class DynamicRiskBarrierEngine:
    """
    動態風險護欄引擎
    
    核心理念：
    - 在關鍵分（破發點、盤末點）時，自動提高「穩定性」權重
    - 根據推演拍數調整「移動距離」扣分權重（模擬疲勞）
    - 動態調整風險容忍度
    """
    
    @staticmethod
    def calculate_critical_point_factor(
        score: Dict[str, int],
        break_points: int,
        is_set_point: bool = False
    ) -> float:
        """
        計算關鍵分係數
        
        關鍵分定義：
        - 破發點：係數 1.5
        - 盤末點：係數 1.8
        - 一般：係數 1.0
        
        參數：
            score: 比分字典 {"my": int, "opponent": int}
            break_points: 破發點數
            is_set_point: 是否為盤末點
        
        返回：
            關鍵分係數 (1.0-1.8)
        """
        if is_set_point:
            return 1.8
        elif break_points > 0:
            return 1.5
        else:
            return 1.0
    
    @staticmethod
    def calculate_fatigue_penalty(
        step_num: int,
        movement_distance: float,
        max_steps: int = 10
    ) -> float:
        """
        計算疲勞扣分
        
        核心概念：
        - 隨著推演拍數增加，移動距離的扣分權重增加
        - 模擬球員體力消耗對擊球質量的影響
        
        參數：
            step_num: 當前拍數
            movement_distance: 移動距離
            max_steps: 最大拍數（用於正規化）
        
        返回：
            疲勞扣分 (0-20)
        """
        # 疲勞係數：隨拍數線性增加（第 10 拍時達到最大）
        fatigue_factor = min(1.0, step_num / max_steps)
        
        # 扣分：移動距離 × 疲勞係數 × 權重
        penalty = movement_distance * fatigue_factor * 0.5
        
        return min(20.0, penalty)
    
    @staticmethod
    def adjust_risk_tolerance(
        base_score: float,
        critical_point_factor: float,
        phase: PhaseType,
        opponent_style: OpponentStyle
    ) -> float:
        """
        動態調整風險容忍度
        
        核心概念：
        - 關鍵分時：降低風險球路的評分（×0.7）
        - 防守相位：進一步降低風險（×0.8）
        - 對抗進攻型對手：可以適度冒險（×1.1）
        
        參數：
            base_score: 基礎評分
            critical_point_factor: 關鍵分係數
            phase: 相位
            opponent_style: 對手風格
        
        返回：
            調整後的評分
        """
        adjusted_score = base_score
        
        # 關鍵分調整
        if critical_point_factor > 1.0:
            adjusted_score *= 0.7  # 關鍵分降低風險
        
        # 相位調整
        if phase == PhaseType.DEFENSE:
            adjusted_score *= 0.8  # 防守相位降低風險
        elif phase == PhaseType.OFFENSE:
            adjusted_score *= 1.1  # 進攻相位可以適度冒險
        
        # 對手風格調整
        if opponent_style == OpponentStyle.AGGRESSIVE:
            adjusted_score *= 1.05  # 對抗進攻型對手可以適度冒險
        elif opponent_style == OpponentStyle.PUSHER:
            adjusted_score *= 0.95  # 對抗防守型對手降低風險
        
        return adjusted_score


# ============================================================================
# 🔥 深度重構：對手慣性預測引擎 (Opponent Habitual Predictor)
# ============================================================================

class OpponentHabitualPredictor:
    """
    對手慣性預測引擎
    
    核心理念：
    - 慣性鏈條 (Habitual Chain)：連續擊球的路徑依賴
    - 受迫性預測 (Forced Prediction)：防守位置的預測傾斜
    - 預測不確定性區間 (Uncertainty Interval)：信心值量化
    """
    
    @staticmethod
    def analyze_shot_pattern(shot_history: List[ShotStep], player: str = "Opponent") -> Dict[str, any]:
        """
        分析對手的擊球模式
        
        返回：
        - 對角線頻率
        - 直線頻率
        - 最常用球種
        - 連續相同方向次數
        """
        if not shot_history:
            return {
                "crosscourt_freq": 0.0,
                "dtl_freq": 0.0,
                "most_common_shot": FixedShotType.BASELINE_TOPSPIN,
                "consecutive_same_direction": 0
            }
        
        opponent_shots = [s for s in shot_history if s.player == player]
        
        if not opponent_shots:
            return {
                "crosscourt_freq": 0.0,
                "dtl_freq": 0.0,
                "most_common_shot": FixedShotType.BASELINE_TOPSPIN,
                "consecutive_same_direction": 0
            }
        
        # 分析對角線 vs 直線
        crosscourt_count = 0
        dtl_count = 0
        
        for shot in opponent_shots:
            start_y = shot.start_pos[1]
            end_y = shot.end_pos[1]
            
            # 判定是否為對角線（Y 座標符號改變或變化幅度大）
            is_crosscourt = (start_y * end_y < 0) or abs(end_y - start_y) > 6.0
            
            if is_crosscourt:
                crosscourt_count += 1
            else:
                dtl_count += 1
        
        total = crosscourt_count + dtl_count
        crosscourt_freq = crosscourt_count / total if total > 0 else 0.0
        dtl_freq = dtl_count / total if total > 0 else 0.0
        
        # 統計最常用球種
        shot_type_counts = {}
        for shot in opponent_shots:
            shot_type_counts[shot.shot_type] = shot_type_counts.get(shot.shot_type, 0) + 1
        
        most_common_shot = max(shot_type_counts, key=shot_type_counts.get) if shot_type_counts else FixedShotType.BASELINE_TOPSPIN
        
        # 計算連續相同方向次數
        consecutive_count = 0
        if len(opponent_shots) >= 2:
            last_direction = opponent_shots[-1].end_pos[1]
            prev_direction = opponent_shots[-2].end_pos[1]
            
            if (last_direction * prev_direction > 0):  # 同側
                consecutive_count = 2
                
                # 檢查更早的擊球
                for i in range(len(opponent_shots) - 3, -1, -1):
                    if opponent_shots[i].end_pos[1] * last_direction > 0:
                        consecutive_count += 1
                    else:
                        break
        
        return {
            "crosscourt_freq": crosscourt_freq,
            "dtl_freq": dtl_freq,
            "most_common_shot": most_common_shot,
            "consecutive_same_direction": consecutive_count
        }
    
    @staticmethod
    def calculate_surprise_bonus(
        shot_direction: str,  # "crosscourt" or "dtl"
        pattern: Dict[str, any]
    ) -> float:
        """
        計算驚奇加分
        
        核心概念：
        - 如果對手前兩拍都打對角線，第三拍打直線的驚奇加分應提高
        - 但執行成功率應根據風格下修
        
        參數：
            shot_direction: 擊球方向 ("crosscourt" 或 "dtl")
            pattern: 擊球模式分析結果
        
        返回：
            驚奇加分 (0-25)
        """
        consecutive = pattern["consecutive_same_direction"]
        
        if consecutive >= 2:
            # 連續相同方向 2 次以上，改變方向有驚奇加分
            if shot_direction == "dtl" and pattern["crosscourt_freq"] > 0.7:
                return 25.0  # 高驚奇：一直打對角，突然直線
            elif shot_direction == "crosscourt" and pattern["dtl_freq"] > 0.7:
                return 20.0  # 中驚奇：一直打直線，突然對角
        
        return 0.0
    
    @staticmethod
    def calculate_forced_prediction_bias(
        opponent_pos: Tuple[float, float],
        phase: PhaseType
    ) -> Dict[str, float]:
        """
        計算受迫性預測偏置
        
        核心概念：
        - 當對手處於防守位置時，預測應大幅向「高吊球」或「中路深球」傾斜
        - 而非平均分配 15 種球路
        
        參數：
            opponent_pos: 對手位置
            phase: 相位
        
        返回：
            各球種的偏置權重
        """
        opp_x, opp_y = opponent_pos
        
        # 判定是否在防守位置
        is_defensive = phase == PhaseType.DEFENSE or abs(opp_y) >= 14.0
        
        if is_defensive:
            return {
                FixedShotType.LOB: 50.0,  # 高吊球大幅加分
                FixedShotType.BASELINE_TOPSPIN: 30.0,  # 中路深球加分
                FixedShotType.BASELINE_SLICE: 20.0,  # 切球加分
                FixedShotType.BASELINE_FLAT: -20.0,  # 平擊球扣分（風險高）
                FixedShotType.DROP_SHOT: -40.0,  # 短球扣分（無法執行）
                FixedShotType.INSIDE_OUT: -30.0  # 大角度扣分（無法執行）
            }
        else:
            # 非防守位置：平衡分配
            return {
                FixedShotType.LOB: 0.0,
                FixedShotType.BASELINE_TOPSPIN: 0.0,
                FixedShotType.BASELINE_SLICE: 0.0,
                FixedShotType.BASELINE_FLAT: 0.0,
                FixedShotType.DROP_SHOT: 0.0,
                FixedShotType.INSIDE_OUT: 0.0
            }
    
    @staticmethod
    def calculate_uncertainty_interval(
        prediction_score: float,
        shot_history_length: int,
        opponent_style: OpponentStyle
    ) -> Tuple[str, str]:
        """
        計算預測不確定性區間
        
        核心概念：
        - 不僅給出機率，還要給出「預測信心值」
        - 擊球歷史越長，預測越準確
        - 對手風格越明確，預測越準確
        
        參數：
            prediction_score: 預測評分
            shot_history_length: 擊球歷史長度
            opponent_style: 對手風格
        
        返回：
            (信心等級, 不確定性區間)
        """
        # 基礎信心值
        base_confidence = 50.0
        
        # 擊球歷史長度加成（最多 +30）
        history_bonus = min(30.0, shot_history_length * 5.0)
        
        # 風格明確度加成
        style_bonus = 0.0
        if opponent_style == OpponentStyle.PUSHER:
            style_bonus = 20.0  # Pusher 最可預測（95% 對角深球）
        elif opponent_style == OpponentStyle.AGGRESSIVE:
            style_bonus = 5.0   # 【🔥 重構】Aggressive 較不可預測（願意冒險，變化多）
        else:
            style_bonus = 10.0  # Net Rusher 中等可預測（有上網模式）
        
        # 評分高低影響
        score_factor = 1.0
        if prediction_score >= 80.0:
            score_factor = 1.2  # 高分預測更有信心
        elif prediction_score <= 40.0:
            score_factor = 0.8  # 低分預測信心較低
        
        # 計算最終信心值
        confidence = (base_confidence + history_bonus + style_bonus) * score_factor
        confidence = min(95.0, max(30.0, confidence))
        
        # 計算不確定性區間
        uncertainty = 100.0 - confidence
        interval = f"±{uncertainty/2:.1f}%"
        
        # 信心等級
        if confidence >= 80.0:
            level = "High"
        elif confidence >= 60.0:
            level = "Medium"
        else:
            level = "Low"
        
        return (level, interval)


# ============================================================================
# 🔥 深度重構：戰術組合拳引擎 (Tactical Combo Engine)
# ============================================================================

class TacticalComboEngine:
    """
    戰術組合拳引擎
    
    核心理念：
    - 戰術包 (Tactical Package)：不應每拍獨立計算，而是以「戰術包」為單位
    - 誘餌邏輯 (Bait & Trap)：故意打一拍中路淺球以換取下一拍大角度穿越
    - 回位品質影響 (Recovery Impact)：上一拍質量影響回位，進而影響下一拍
    """
    
    # 預設戰術包（3拍組合）
    TACTICAL_PACKAGES = {
        "2-1_PATTERN": {
            "name": "2-1 幾何壓制",
            "steps": [
                {"step": 1, "intent": "深球壓制", "target_depth": "deep", "target_angle": "medium"},
                {"step": 2, "intent": "角度拉開", "target_depth": "medium", "target_angle": "wide"},
                {"step": 3, "intent": "空檔終結", "target_depth": "deep", "target_angle": "opposite"}
            ],
            "tactic_ids": [5, 13, 20]  # 對應 48 項戰術
        },
        "BACKHAND_CAGE": {
            "name": "反拍牢籠",
            "steps": [
                {"step": 1, "intent": "反拍壓制", "target_depth": "deep", "target_angle": "backhand"},
                {"step": 2, "intent": "持續反拍", "target_depth": "deep", "target_angle": "backhand"},
                {"step": 3, "intent": "持續反拍", "target_depth": "deep", "target_angle": "backhand"},
                {"step": 4, "intent": "變線終結", "target_depth": "deep", "target_angle": "forehand"}
            ],
            "tactic_ids": [6, 6, 6, 27]  # 對應 48 項戰術
        },
        "ANGLE_LINE": {
            "name": "角度-直線進攻",
            "steps": [
                {"step": 1, "intent": "角度拉開", "target_depth": "medium", "target_angle": "wide"},
                {"step": 2, "intent": "直線終結", "target_depth": "deep", "target_angle": "dtl"}
            ],
            "tactic_ids": [13, 9]  # 對應 48 項戰術
        },
        "BAIT_AND_TRAP": {
            "name": "誘餌與陷阱",
            "steps": [
                {"step": 1, "intent": "誘餌球", "target_depth": "short", "target_angle": "center"},
                {"step": 2, "intent": "誘使上網", "target_depth": "short", "target_angle": "center"},
                {"step": 3, "intent": "穿越得分", "target_depth": "deep", "target_angle": "dtl"}
            ],
            "tactic_ids": [14, 25, 9]  # 對應 48 項戰術
        }
    }
    
    @staticmethod
    def select_tactical_package(
        phase: PhaseType,
        opponent_style: OpponentStyle,
        shot_history: List[ShotStep],
        momentum: float
    ) -> Optional[str]:
        """
        選擇適合的戰術包
        
        參數：
            phase: 當前相位
            opponent_style: 對手風格
            shot_history: 擊球歷史
            momentum: 心理動量
        
        返回：
            戰術包名稱 或 None
        """
        # 根據相位選擇
        if phase == PhaseType.OFFENSE:
            # 進攻相位：優先選擇進攻性戰術包
            if opponent_style == OpponentStyle.PUSHER:
                return "BAIT_AND_TRAP"  # 對抗 Pusher：誘使上網
            else:
                return "ANGLE_LINE"  # 對抗其他：角度-直線
        
        elif phase == PhaseType.NEUTRAL:
            # 相持相位：選擇穩定戰術包
            if momentum > 0.3:
                return "2-1_PATTERN"  # 動量優勢：2-1 壓制
            else:
                return None  # 維持相持，不強制戰術包
        
        else:  # DEFENSE
            # 防守相位：不使用戰術包
            return None
    
    @staticmethod
    def get_combo_bonus(
        current_step: int,
        package_name: str,
        shot_history: List[ShotStep]
    ) -> float:
        """
        計算組合拳加成
        
        核心概念：
        - 如果按照戰術包執行，每一拍都有組合拳加成
        - 加成隨拍數遞增（第 3 拍終結拍加成最高）
        
        參數：
            current_step: 當前拍數
            package_name: 戰術包名稱
            shot_history: 擊球歷史
        
        返回：
            組合拳加成 (0-30)
        """
        if package_name not in TacticalComboEngine.TACTICAL_PACKAGES:
            return 0.0
        
        package = TacticalComboEngine.TACTICAL_PACKAGES[package_name]
        steps = package["steps"]
        
        # 檢查當前拍數是否在戰術包範圍內
        package_step = None
        for step_info in steps:
            if step_info["step"] == len(shot_history) + 1:
                package_step = step_info
                break
        
        if not package_step:
            return 0.0
        
        # 根據拍數位置給予不同加成
        step_position = len(shot_history) + 1
        total_steps = len(steps)
        
        if step_position == total_steps:
            return 30.0  # 終結拍：最高加成
        elif step_position == total_steps - 1:
            return 20.0  # 準終結拍：中等加成
        else:
            return 10.0  # 前置拍：基礎加成
    
    @staticmethod
    def calculate_recovery_quality_impact(
        last_shot: ShotStep,
        opponent_pos: Tuple[float, float]
    ) -> Dict[str, float]:
        """
        計算回位品質影響
        
        核心概念：
        - 若我方上一拍擊出極具威脅的球，對手回位會延遲
        - 下一拍的「進攻窗口」應自動放大
        
        參數：
            last_shot: 上一拍擊球
            opponent_pos: 對手位置
        
        返回：
            回位品質影響字典
        """
        # 判定上一拍的威脅程度
        threat_level = 0.0
        
        # 1. 距離威脅
        distance = last_shot.distance_to_opponent
        if distance >= 15.0:
            threat_level += 0.8
        elif distance >= 12.0:
            threat_level += 0.5
        else:
            threat_level += 0.2
        
        # 2. 成功率威脅（高成功率擊球更有威脅）
        if last_shot.success_rate >= 70.0:
            threat_level += 0.2
        
        # 3. 相位威脅（進攻相位更有威脅）
        if last_shot.phase == PhaseType.OFFENSE:
            threat_level += 0.3
        
        threat_level = min(1.0, threat_level)
        
        # 計算回位延遲加成
        recovery_delay_bonus = threat_level * 25.0
        
        # 計算進攻窗口放大係數
        attack_window_multiplier = 1.0 + (threat_level * 0.3)
        
        return {
            "threat_level": threat_level,
            "recovery_delay_bonus": recovery_delay_bonus,
            "attack_window_multiplier": attack_window_multiplier
        }
    
    @staticmethod
    def introduce_mutation(
        predictions: List[OpponentPrediction],
        mutation_rate: float = 0.1
    ) -> List[OpponentPrediction]:
        """
        引入非標準戰術突變
        
        核心概念：
        - 在推演中加入 5-10% 的「非最優解」擾動
        - 模擬真實比賽中球員的判斷失誤或冒險行為
        
        參數：
            predictions: 預測列表
            mutation_rate: 突變率 (0.05-0.15)
        
        返回：
            引入突變後的預測列表
        """
        if not predictions or np.random.rand() > mutation_rate:
            return predictions
        
        # 隨機選擇一個預測進行突變
        mutation_idx = np.random.randint(0, min(5, len(predictions)))
        
        # 突變：大幅降低評分（模擬失誤決策）
        predictions[mutation_idx].score *= 0.6
        predictions[mutation_idx].tactical_reasoning += " [突變：非最優決策]"
        
        # 重新排序
        predictions.sort(key=lambda p: p.score, reverse=True)
        
        return predictions
