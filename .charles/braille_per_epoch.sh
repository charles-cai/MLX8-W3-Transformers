#!/bin/bash

# braille_per_epoch.sh - Capture and playback colored terminal output from transformer training
# Usage: 
#   ./braille_per_epoch.sh capture <session_name> [command]
#   ./braille_per_epoch.sh playback <session_name>
#   ./braille_per_epoch.sh list
#   ./braille_per_epoch.sh clean [session_name]

set -e

# Configuration
RECORDINGS_DIR=".data/terminal_recordings"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Colors for script output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Create recordings directory
mkdir -p "$RECORDINGS_DIR"

print_usage() {
    echo -e "${CYAN}braille_per_epoch.sh - Terminal Recording Tool${NC}"
    echo ""
    echo -e "${YELLOW}Usage:${NC}"
    echo "  $0 capture <session_name> [command]  - Record terminal session with colors"
    echo "  $0 playback <session_name>           - Playback recorded session"
    echo "  $0 list                              - List all recorded sessions"
    echo "  $0 clean [session_name]              - Clean recordings (all or specific)"
    echo ""
    echo -e "${YELLOW}Examples:${NC}"
    echo "  $0 capture encoder_decoder_epoch5 python encoder_decoder_models.py"
    echo "  $0 capture vit_training uv run python encoder_only_models.py"
    echo "  $0 capture viz_analysis python viz.py"
    echo "  $0 playback encoder_decoder_epoch5"
    echo ""
    echo -e "${YELLOW}Notes:${NC}"
    echo "  - Recordings are saved in $RECORDINGS_DIR"
    echo "  - Use 'script' command for full terminal capture with colors"
    echo "  - Playback uses 'cat' to preserve colors and formatting"
}

capture_session() {
    local session_name="$1"
    shift
    local command="$*"
    
    if [[ -z "$session_name" ]]; then
        echo -e "${RED}Error: Session name required${NC}"
        print_usage
        exit 1
    fi
    
    local recording_file="$RECORDINGS_DIR/${session_name}_${TIMESTAMP}.txt"
    local info_file="$RECORDINGS_DIR/${session_name}_${TIMESTAMP}.info"
    
    echo -e "${GREEN}Starting recording session: ${session_name}${NC}"
    echo -e "${BLUE}Recording file: ${recording_file}${NC}"
    echo -e "${BLUE}Command: ${command:-'interactive shell'}${NC}"
    echo ""
    
    # Save session info
    cat > "$info_file" << EOF
session_name: $session_name
timestamp: $TIMESTAMP
date: $(date)
command: ${command:-'interactive shell'}
working_directory: $(pwd)
user: $(whoami)
hostname: $(hostname)
EOF
    
    if [[ -n "$command" ]]; then
        echo -e "${YELLOW}Choose recording mode:${NC}"
        echo "  1) Full capture (all output including progress bars)"
        echo "  2) Clean capture (reduced tqdm noise)"
        echo -n "Enter choice [1-2]: "
        read -r mode
        
        echo ""
        echo -e "${YELLOW}Recording command execution...${NC}"
        echo -e "${PURPLE}Press Ctrl+C to stop recording${NC}"
        echo ""
        
        case "$mode" in
            2)
                # Clean capture mode - filter tqdm output during recording
                script -q -c "$command" /dev/stdout | \
                    stdbuf -oL sed 's/\r[^⠀-⣿┌┐└┘│─]*\r/\r/g' | \
                    tee "$recording_file"
                ;;
            *)
                # Full capture mode (default)
                script -q -c "$command" "$recording_file"
                ;;
        esac
    else
        echo -e "${YELLOW}Starting interactive recording session...${NC}"
        echo -e "${PURPLE}Type 'exit' to stop recording${NC}"
        echo ""
        
        # Interactive session
        script -q "$recording_file"
    fi
    
    echo ""
    echo -e "${GREEN}Recording saved: ${recording_file}${NC}"
    echo -e "${BLUE}Session info: ${info_file}${NC}"
    
    # Show file size
    local size=$(du -h "$recording_file" | cut -f1)
    echo -e "${CYAN}Recording size: ${size}${NC}"
}

playback_session() {
    local session_name="$1"
    
    if [[ -z "$session_name" ]]; then
        echo -e "${RED}Error: Session name required${NC}"
        print_usage
        exit 1
    fi
    
    # Find the most recent recording for this session
    local recording_file
    recording_file=$(find "$RECORDINGS_DIR" -name "${session_name}_*.txt" -type f | sort -r | head -n1)
    
    if [[ -z "$recording_file" || ! -f "$recording_file" ]]; then
        echo -e "${RED}Error: No recording found for session '${session_name}'${NC}"
        echo -e "${YELLOW}Available sessions:${NC}"
        list_sessions
        exit 1
    fi
    
    local info_file="${recording_file%.txt}.info"
    
    echo -e "${GREEN}Playing back session: ${session_name}${NC}"
    echo -e "${BLUE}Recording file: ${recording_file}${NC}"
    
    if [[ -f "$info_file" ]]; then
        echo -e "${CYAN}Session info:${NC}"
        cat "$info_file" | sed 's/^/  /'
    fi
    
    echo ""
    echo -e "${YELLOW}Choose playback mode:${NC}"
    echo "  1) Raw playback (all characters, colors, progress bars)"
    echo "  2) Clean playback (filter progress bars, keep colors)"
    echo "  3) Text only (no colors, no progress bars)"
    echo "  4) Braille only (extract only braille displays)"
    echo -n "Enter choice [1-4]: "
    read -r choice
    
    echo ""
    echo -e "${PURPLE}===========================================${NC}"
    
    case "$choice" in
        1)
            echo -e "${YELLOW}Raw playback mode...${NC}"
            _playback_raw "$recording_file"
            ;;
        2)
            echo -e "${YELLOW}Clean playback mode (filtering progress bars)...${NC}"
            _playback_clean "$recording_file"
            ;;
        3)
            echo -e "${YELLOW}Text-only mode (no colors)...${NC}"
            _playback_text_only "$recording_file"
            ;;
        4)
            echo -e "${YELLOW}Braille-only mode...${NC}"
            _playback_braille_only "$recording_file"
            ;;
        *)
            echo -e "${YELLOW}Invalid choice, using clean mode...${NC}"
            _playback_clean "$recording_file"
            ;;
    esac
    
    echo -e "${PURPLE}===========================================${NC}"
    echo -e "${GREEN}Playback completed${NC}"
}

_playback_raw() {
    local recording_file="$1"
    local lines=$(wc -l < "$recording_file")
    
    if [[ $lines -gt $(tput lines) ]]; then
        echo -e "${YELLOW}Output is long ($lines lines), using pager. Press 'q' to quit.${NC}"
        sleep 2
        cat "$recording_file" | less -R
    else
        cat "$recording_file"
    fi
}

_playback_clean() {
    local recording_file="$1"
    
    # Filter out problematic sequences while keeping colors and braille
    cat "$recording_file" | \
        # Remove carriage returns that overwrite lines
        tr -d '\r' | \
        # Remove lines that are mostly progress bar characters
        grep -v '█\{20,\}' | \
        # Remove lines with excessive spaces and progress indicators
        grep -v '^[[:space:]]*[0-9]*%.*█.*batch/s' | \
        # Remove tqdm progress lines but keep other content
        grep -v 'Epoch [0-9]*/[0-9]*:.*%.*█.*\[.*<.*batch/s' | \
        # Remove excessive blank lines (keep max 2 consecutive)
        cat -s | \
        # Page if output is long
        if [[ $(wc -l) -gt $(tput lines) ]]; then
            less -R
        else
            cat
        fi
}

_playback_text_only() {
    local recording_file="$1"
    
    # Strip all ANSI escape sequences and clean up
    cat "$recording_file" | \
        # Remove all ANSI escape sequences (colors, cursor movements, etc.)
        sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | \
        # Remove carriage returns
        tr -d '\r' | \
        # Remove progress bar lines
        grep -v '█\{10,\}' | \
        grep -v '^[[:space:]]*[0-9]*%.*batch/s' | \
        # Remove excessive whitespace
        sed 's/[[:space:]]\{10,\}/ /g' | \
        # Remove excessive blank lines
        cat -s | \
        # Page if long
        if [[ $(wc -l) -gt $(tput lines) ]]; then
            less
        else
            cat
        fi
}

_playback_braille_only() {
    local recording_file="$1"
    
    # Extract only braille-related content
    echo -e "${CYAN}Extracting braille displays...${NC}"
    echo ""
    
    cat "$recording_file" | \
        # Remove carriage returns
        tr -d '\r' | \
        # Extract blocks that contain braille patterns
        awk '
        BEGIN { in_braille = 0; buffer = ""; }
        
        # Start of braille section markers
        /EPOCH.*VALIDATION|Legend:.*Expected.*Predicted|Epoch.*Batch Display/ { 
            in_braille = 1; 
            if (buffer) print buffer "\n"; 
            buffer = $0; 
            next; 
        }
        
        # Lines with braille characters or box drawing
        /[⠀-⣿]|[┌┐└┘│─]/ { 
            if (in_braille) buffer = buffer "\n" $0; 
            next; 
        }
        
        # Lines with color codes for predictions
        /\[9[0-7]m[0-9]\[0m/ { 
            if (in_braille) buffer = buffer "\n" $0; 
            next; 
        }
        
        # Empty lines in braille sections
        /^[[:space:]]*$/ { 
            if (in_braille) buffer = buffer "\n" $0; 
            next; 
        }
        
        # Section separators
        /^=+$/ { 
            if (in_braille) {
                buffer = buffer "\n" $0;
                print buffer "\n";
                buffer = "";
                in_braille = 0;
            }
            next;
        }
        
        # Reset if we hit other content
        /Epoch [0-9]*:.*Loss:.*Acc:/ { 
            if (in_braille && buffer) {
                print buffer "\n";
                buffer = "";
            }
            in_braille = 0; 
        }
        
        END { if (buffer) print buffer; }
        ' | \
        # Clean up excessive blank lines
        cat -s | \
        # Page if long
        if [[ $(wc -l) -gt $(tput lines) ]]; then
            less -R
        else
            cat
        fi
}

# Quick capture functions for common scenarios
quick_encoder_decoder() {
    capture_session "encoder_decoder_$(date +%m%d_%H%M)" "uv run python encoder_decoder_models.py"
}

quick_encoder_only() {
    capture_session "encoder_only_$(date +%m%d_%H%M)" "uv run python encoder_only_models.py"
}

quick_viz() {
    capture_session "viz_analysis_$(date +%m%d_%H%M)" "uv run python viz.py"
}

# Main script logic
case "${1:-}" in
    capture)
        shift
        capture_session "$@"
        ;;
    playback|play)
        shift
        playback_session "$@"
        ;;
    list|ls)
        list_sessions
        ;;
    clean|rm)
        shift
        clean_sessions "$@"
        ;;
    quick-enc-dec)
        quick_encoder_decoder
        ;;
    quick-enc-only)
        quick_encoder_only
        ;;
    quick-viz)
        quick_viz
        ;;
    help|--help|-h)
        print_usage
        ;;
    "")
        echo -e "${RED}Error: No command specified${NC}"
        echo ""
        print_usage
        exit 1
        ;;
    *)
        echo -e "${RED}Error: Unknown command '${1}'${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac
