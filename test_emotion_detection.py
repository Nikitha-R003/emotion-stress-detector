#!/usr/bin/env python3
"""
Test script for emotion detection functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import detect_emotion, estimate_stress_level

def test_emotion_detection():
    """Test the emotion detection with various inputs"""

    test_cases = [
        # Sleepy/tired keywords
        ("I'm feeling so sleepy today", "sleepy"),
        ("I'm exhausted and need a nap", "sleepy"),
        ("I'm tired after work", "sleepy"),
        ("I feel drowsy", "sleepy"),

        # Fear/stressed
        ("I'm really scared about the presentation", "stressed"),
        ("I'm anxious about tomorrow", "stressed"),
        ("I'm worried about the future", "stressed"),

        # Happy
        ("I'm so happy today!", "happy"),
        ("This is amazing!", "happy"),

        # Sad
        ("I'm feeling really sad", "sad"),
        ("I'm depressed", "sad"),

        # Angry
        ("I'm furious about this", "angry"),
        ("This makes me so angry", "angry"),

        # Calm
        ("I'm feeling peaceful", "calm"),
        ("Everything is fine", "calm"),

        # Surprised
        ("Wow, I can't believe it!", "surprised"),
        ("That's unexpected", "surprised"),

        # Disgusted
        ("This is disgusting", "disgusted"),
        ("I hate this", "disgusted"),
    ]

    print("Testing Emotion Detection...")
    print("=" * 50)

    passed = 0
    total = len(test_cases)

    for text, expected in test_cases:
        try:
            emotion, confidence = detect_emotion(text)
            stress_level = estimate_stress_level(emotion)

            status = "✓" if emotion == expected else "✗"
            if emotion == expected:
                passed += 1

            print(f"{status} '{text}' -> {emotion} (confidence: {confidence:.2f}) | stress: {stress_level}")
            if emotion != expected:
                print(f"   Expected: {expected}, Got: {emotion}")

        except Exception as e:
            print(f"✗ Error with '{text}': {str(e)}")

    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")

    # Test edge cases
    print("\nTesting Edge Cases...")
    edge_cases = [
        ("", "neutral"),  # Empty text
        ("I feel okay", "calm"),  # Neutral feeling
        ("Mixed feelings about this", "neutral"),  # Mixed emotions
    ]

    for text, expected in edge_cases:
        try:
            emotion, confidence = detect_emotion(text)
            print(f"Edge: '{text}' -> {emotion} (expected: {expected})")
        except Exception as e:
            print(f"Edge Error: '{text}' -> {str(e)}")

    return passed == total

if __name__ == "__main__":
    success = test_emotion_detection()
    sys.exit(0 if success else 1)
