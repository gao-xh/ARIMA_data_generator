import sys
from pathlib import Path
import logging

# Add src to python path so we can import modules
sys.path.append(str(Path(__file__).parent))

try:
    from src.data_simulator import DataSimulator
except ImportError as e:
    print(f"Standard import failed: {e}. Trying raw import.")
    import src.data_simulator
    DataSimulator = src.data_simulator.DataSimulator

def main():
    print("Starting Synthetic Dataset Generation...")
    
    try:
        sim = DataSimulator()
        sim.load_base_data()
        sim.simulate()
        sim.save(Path("data/processed/synthetic_sales.csv"))
        
        print("\nDataset Generation Complete!")
        print(f"Data saved to data/processed/synthetic_sales.csv")
        print(f"Simulated {len(sim.simulation_data)} records for {len(sim.clinics)} clinics over the 2024-2025 period.")
            
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
