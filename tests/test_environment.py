# tests/test_environment.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.base_config import BaseConfig
from environment.battle_env import BattleEnvironment

def test_environment():
    """Тест среды"""
    print("🧪 Testing Battle Environment...")
    
    config = BaseConfig()
    config.map_size = 32
    config.max_units = 5
    config.max_steps = 50
    
    env = BattleEnvironment(config)
    state = env.reset()
    
    print(f"✅ Environment initialized")
    print(f"   Map size: {config.map_size}x{config.map_size}")
    print(f"   Attacker units: {len(env.attacker_units)}")
    print(f"   Defender units: {len(env.defender_units)}")
    
    # Тест шага
    for i in range(10):
        attacker_actions = {}
        defender_actions = {}
        
        for unit in env.attacker_units:
            attacker_actions[str(unit.id)] = {
                'move': (unit.x + 2, unit.y)
            }
        
        next_state, rewards, done, info = env.step(attacker_actions, defender_actions)
        print(f"   Step {i}: Att={rewards['attacker']:.1f}, Def={rewards['defender']:.1f}")
        
        if done:
            break
    
    print("✅ Test completed!")

if __name__ == "__main__":
    test_environment()