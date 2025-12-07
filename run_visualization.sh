#!/bin/bash
# Convenience script to run belief visualizations

echo "=========================================="
echo "  Belief Visualization Runner"
echo "=========================================="
echo ""

# Default values
MODE="quick"
STEPS=10
SCENE_ID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="quick"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --steps)
            STEPS="$2"
            shift 2
            ;;
        --scene)
            SCENE_ID="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run_visualization.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --quick              Run quick visualization (default)"
            echo "  --full               Run full Figure 1 style visualization"
            echo "  --steps N            Number of steps (default: 10 for quick, 20 for full)"
            echo "  --scene N            Scene ID to load (e.g., 5)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  ./run_visualization.sh --quick --steps 15"
            echo "  ./run_visualization.sh --full --scene 5 --steps 25"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Set default steps based on mode
if [ "$MODE" = "full" ] && [ "$STEPS" = "10" ]; then
    STEPS=20
fi

echo "Mode: $MODE"
echo "Steps: $STEPS"
if [ -n "$SCENE_ID" ]; then
    echo "Scene ID: $SCENE_ID"
fi
echo ""

# Run the appropriate script
if [ "$MODE" = "quick" ]; then
    echo "Running quick visualization..."
    python quick_visualize.py --steps $STEPS

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "✓ Success! View results in ./quick_viz/"
        echo "  Main output: ./quick_viz/summary.png"
    fi

elif [ "$MODE" = "full" ]; then
    echo "Running full visualization..."

    CMD="python visualize_belief_figure1.py --steps $STEPS"

    if [ -n "$SCENE_ID" ]; then
        CMD="$CMD --scene-id $SCENE_ID"
    fi

    eval $CMD

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        # Find the most recent output directory
        OUTPUT_DIR=$(ls -td belief_visualizations_* | head -1)
        echo ""
        echo "✓ Success! View results in ./$OUTPUT_DIR/"
        echo "  Main outputs:"
        echo "    - ./$OUTPUT_DIR/figure1_style_comparison.png"
        echo "    - ./$OUTPUT_DIR/extended_comparison.png"
        echo ""
        echo "To create a video:"
        echo "  cd $OUTPUT_DIR"
        echo "  ffmpeg -framerate 2 -pattern_type glob -i 'frame_*.png' \\"
        echo "         -c:v libx264 -pix_fmt yuv420p belief_evolution.mp4"
    fi
fi

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "✗ Error occurred during visualization"
    exit $EXIT_CODE
fi

echo ""
echo "=========================================="
echo "  Done!"
echo "=========================================="
