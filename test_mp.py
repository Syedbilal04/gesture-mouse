import mediapipe as mp
try:
    import mediapipe.python.solutions
    print("Imported mediapipe.python.solutions")
except ImportError as e:
    print("Failed to import mediapipe.python.solutions:", e)

try:
    from mediapipe import solutions
    print("Imported solutions from mediapipe")
except ImportError as e:
    print("Failed to import solutions from mediapipe:", e)
