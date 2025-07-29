# omniparser-autogui-mcp

（[日本語版はこちら](README_ja.md)）

This is an [MCP server](https://modelcontextprotocol.io/introduction) that analyzes the screen with [OmniParser](https://github.com/microsoft/OmniParser) and automatically operates the GUI with advanced keyboard/mouse automation and macro support.
Confirmed on Windows.

## ✨ New Features

### 🎯 Complete Keyboard & Mouse Operations
- **Precise coordinate control** - Click, move, drag at exact positions
- **Human-like input simulation** - Natural movement curves and random delays
- **Advanced keyboard operations** - Complex key combinations and text input
- **Enhanced scrolling** - Directional scrolling with position control

### 🤖 Macro System
- **Record & Playback** - Capture complex operation sequences
- **Macro Management** - Create, edit, delete, and organize macros
- **Predefined Templates** - Common operations like copy/paste, save, refresh
- **Repeat Control** - Loop macros with customizable delays

### 🎮 Gaming Features
- **Pixel Detection** - Read and monitor pixel colors
- **Rapid Clicking** - High-speed clicking for games
- **Combo Sequences** - Execute complex skill combinations
- **Anti-Detection** - Human-like timing to avoid bot detection

### 🖥️ Window Management
- **Window Enumeration** - List and switch between windows
- **Region Screenshots** - Capture specific screen areas
- **Multi-monitor Support** - Work across multiple displays
- **Focus Control** - Automatic window activation

### 📋 Available MCP Tools

#### Screen Analysis
- `omniparser_details_on_screen()` - AI-powered screen analysis

#### Basic Operations (Enhanced)
- `omniparser_click()` - Element-based clicking
- `omniparser_drags()` - Element-based dragging
- `omniparser_write()` - Text input with element targeting
- `omniparser_scroll()` - Basic scrolling

#### Coordinate-Based Operations (New)
- `mouse_click_coordinate()` - Precise coordinate clicking
- `mouse_move_coordinate()` - Smooth mouse movement
- `mouse_drag_coordinate()` - Coordinate-based dragging
- `get_mouse_position()` - Current mouse position

#### Advanced Keyboard (New)
- `keyboard_type_text()` - Human-like text input
- `keyboard_press_keys()` - Multi-key combinations
- `keyboard_hotkey()` - Shortcut key execution
- `scroll_advanced()` - Directional scrolling

#### Macro System (New)
- `macro_start_recording()` - Begin macro recording
- `macro_stop_recording()` - End macro recording
- `macro_play()` - Execute recorded macros
- `macro_list()` - List available macros
- `macro_delete()` - Remove macros
- `macro_get_info()` - Macro details

#### Gaming Features (New)
- `get_pixel_color()` - Read pixel RGB values
- `wait_for_pixel_color()` - Wait for color changes
- `rapid_click()` - High-speed clicking
- `combo_sequence()` - Execute skill combos

#### Window Management (New)
- `list_windows()` - Enumerate all windows
- `switch_to_window()` - Change active window
- `get_screen_size()` - Screen dimensions
- `take_screenshot_region()` - Partial screenshots

#### Utilities (New)
- `create_predefined_macro()` - Quick macro templates
- `get_system_info()` - System status information

See [ADVANCED_USAGE.md](ADVANCED_USAGE.md) for detailed usage examples and best practices.

## License notes

This is MIT license, but Excluding submodules and sub packages.  
OmniParser's repository is CC-BY-4.0.  
Each OmniParser model has a different license ([reference](https://github.com/microsoft/OmniParser?tab=readme-ov-file#model-weights-license)).

## Installation

1. Please do the following:

```
git clone --recursive https://github.com/NON906/omniparser-autogui-mcp.git
cd omniparser-autogui-mcp
uv sync
set OCR_LANG=en
uv run download_models.py
```

(Other than Windows, use ``export`` instead of ``set``.)  
(If you want ``langchain_example.py`` to work, ``uv sync --extra langchain`` instead.)

2. Add this to your ``claude_desktop_config.json``:

```claude_desktop_config.json
{
  "mcpServers": {
    "omniparser_autogui_mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "D:\\CLONED_PATH\\omniparser-autogui-mcp",
        "run",
        "omniparser-autogui-mcp"
      ],
      "env": {
        "PYTHONIOENCODING": "utf-8",
        "OCR_LANG": "en"
      }
    }
  }
}
```

(Replace ``D:\\CLONED_PATH\\omniparser-autogui-mcp`` with the directory you cloned.)

``env`` allows for the following additional configurations:

- ``OMNI_PARSER_BACKEND_LOAD``  
If it does not work with other clients (such as [LibreChat](https://github.com/danny-avila/LibreChat)), specify ``1``.

- ``TARGET_WINDOW_NAME``  
If you want to specify the window to operate, please specify the window name.  
If not specified, operates on the entire screen.

- ``OMNI_PARSER_SERVER``  
If you want OmniParser processing to be done on another device, specify the server's address and port, such as ``127.0.0.1:8000``.  
The server can be started with ``uv run omniparserserver``.

- ``SSE_HOST``, ``SSE_PORT``  
If specified, communication will be done via SSE instead of stdio.

- ``SOM_MODEL_PATH``, ``CAPTION_MODEL_NAME``, ``CAPTION_MODEL_PATH``, ``OMNI_PARSER_DEVICE``, ``BOX_TRESHOLD``  
These are for OmniParser configuration.  
Usually, they are not necessary.

## Usage Examples

- Search for "MCP server" in the on-screen browser.

etc.