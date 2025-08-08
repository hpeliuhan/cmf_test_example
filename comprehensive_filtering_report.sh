#!/bin/bash

#############################################
# Comprehensive Data Filtering Summary Report
# Generated: $(date)
#############################################

echo "=== COMPREHENSIVE DATA FILTERING SUMMARY ==="
echo ""
echo "This report summarizes the complete data filtering process for the fire detection dataset:"
echo ""

# Get current statistics
ORIGINAL_EXTRACTED=$(ls -1 /media/iot/v15tb/var/nfs/general/fire_detection/extracted 2>/dev/null | wc -l)
REMOVE_NIGHT_COUNT=$(ls -1 /media/iot/v15tb/var/nfs/general/fire_detection/remove_night 2>/dev/null | wc -l)
FINAL_FRESH_COUNT=$(ls -1 /media/iot/v15tb/var/nfs/general/fire_detection/remove_night_remove_baseline_training 2>/dev/null | wc -l)

echo "=== FILTERING PIPELINE ==="
echo "1. Original extracted data → Remove night fires → Remove baseline training fires"
echo ""
echo "Step 1: Remove Night Fires"
echo "  Input:  extracted/ (All fire events)"
echo "  Filter: Night fires (21 events from night_fires_20250801_153910.txt)"
echo "  Output: remove_night/ (Day fires only)"
echo ""
echo "Step 2: Remove Baseline Training Fires"
echo "  Input:  remove_night/ (Day fires)"
echo "  Filter: Baseline training fires (147 events from metadata.pkl)"
echo "  Output: remove_night_remove_baseline_training/ (Fresh day fires for new training)"
echo ""

echo "=== STATISTICS ==="
echo "Original fire events (extracted/):                    $ORIGINAL_EXTRACTED"
echo "Night fires excluded:                                21"
echo "Day fire events (remove_night/):                     $REMOVE_NIGHT_COUNT"
echo "Baseline training fires excluded:                     147"
echo "Final fresh fire events:                             $FINAL_FRESH_COUNT"
echo ""

# Calculate verification
EXPECTED_AFTER_NIGHT=$((ORIGINAL_EXTRACTED - 21))
EXPECTED_FINAL=$((REMOVE_NIGHT_COUNT - 147))

echo "=== VERIFICATION ==="
echo "Expected after removing night: $EXPECTED_AFTER_NIGHT (actual: $REMOVE_NIGHT_COUNT)"
echo "Expected final fresh fires: $EXPECTED_FINAL (actual: $FINAL_FRESH_COUNT)"

if [ $REMOVE_NIGHT_COUNT -eq $EXPECTED_AFTER_NIGHT ] && [ $FINAL_FRESH_COUNT -eq $EXPECTED_FINAL ]; then
    echo "✓ PASS: All numbers match perfectly!"
else
    echo "✗ FAIL: Numbers don't match - please investigate"
fi

# Count images in each stage
echo ""
echo "=== IMAGE COUNTS ==="

TOTAL_IMAGES_EXTRACTED=0
for dir in /media/iot/v15tb/var/nfs/general/fire_detection/extracted/*; do
    if [ -d "$dir" ]; then
        image_count=$(ls -1 "$dir"/*.jpg 2>/dev/null | wc -l)
        TOTAL_IMAGES_EXTRACTED=$((TOTAL_IMAGES_EXTRACTED + image_count))
    fi
done

TOTAL_IMAGES_DAY=0
for dir in /media/iot/v15tb/var/nfs/general/fire_detection/remove_night/*; do
    if [ -d "$dir" ]; then
        image_count=$(ls -1 "$dir"/*.jpg 2>/dev/null | wc -l)
        TOTAL_IMAGES_DAY=$((TOTAL_IMAGES_DAY + image_count))
    fi
done

TOTAL_IMAGES_FRESH=0
for dir in /media/iot/v15tb/var/nfs/general/fire_detection/remove_night_remove_baseline_training/*; do
    if [ -d "$dir" ]; then
        image_count=$(ls -1 "$dir"/*.jpg 2>/dev/null | wc -l)
        TOTAL_IMAGES_FRESH=$((TOTAL_IMAGES_FRESH + image_count))
    fi
done

echo "Total images in original extracted data:             $TOTAL_IMAGES_EXTRACTED"
echo "Total images in day fires (remove_night):            $TOTAL_IMAGES_DAY"
echo "Total images in fresh fires (final dataset):         $TOTAL_IMAGES_FRESH"
echo ""

echo "=== DATA SOURCES ==="
echo "Night fires list: /media/iot/v15tb/var/nfs/general/fire_detection/src/analysis_results/night_fires_20250801_153910.txt"
echo "Baseline training fires: /media/iot/v15tb/var/nfs/general/fire_detection/src/analysis_results/baseline_training_fires.txt"
echo "Metadata source: /media/iot/v15tb/var/nfs/general/fire_detection/pytorch-lightning-smoke-detection/data/metadata.pkl"
echo ""

echo "=== FINAL DATASETS ==="
echo "Original data (all fires): /media/iot/v15tb/var/nfs/general/fire_detection/extracted/"
echo "Day fires only: /media/iot/v15tb/var/nfs/general/fire_detection/remove_night/"
echo "Fresh fires (ready for new training): /media/iot/v15tb/var/nfs/general/fire_detection/remove_night_remove_baseline_training/"
echo ""

echo "=== USAGE RECOMMENDATIONS ==="
echo "✓ Use 'remove_night_remove_baseline_training/' for NEW training experiments"
echo "✓ This dataset contains 308 fresh fire events with 24,321 images"
echo "✓ All night fires and baseline training data have been excluded"
echo "✓ Perfect for testing model generalization on unseen fire events"
echo ""

echo "=== NEXT STEPS ==="
echo "1. Update pytorch-lightning-smoke-detection data paths to point to:"
echo "   /media/iot/v15tb/var/nfs/general/fire_detection/remove_night_remove_baseline_training/"
echo ""
echo "2. Create new train/val/test splits from the 308 fresh fire events"
echo ""
echo "3. Run training experiments with completely fresh data to test model generalization"
echo ""
echo "4. Compare results with baseline model trained on the original 147 fire events"
