# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a WeChat Mini Program (微信小程序) using the Skyline renderer and glass-easel component framework.

## Development

### Opening the Project
Open this project in WeChat DevTools (微信开发者工具). The AppID is configured in `project.config.json`.

### Linting
Run ESLint to check code quality. The configuration is in `.eslintrc.js` and includes WeChat-specific globals (`wx`, `App`, `Page`, `Component`, etc.).

### Testing
Use WeChat DevTools simulator for testing. The project uses Skyline renderer which requires SDK version 3.0.0+.

## Architecture

### Rendering Engine
- **Renderer**: Skyline (configured in `app.json`)
- **Component Framework**: glass-easel
- **Navigation**: Custom navigation bar (set via `navigationStyle: "custom"`)

### File Structure
WeChat Mini Program follows a specific structure where each page/component has four files:
- `.js` - Logic
- `.wxml` - Template (markup)
- `.wxss` - Styles
- `.json` - Configuration

### Key Components

#### navigation-bar
Custom navigation bar component located in `components/navigation-bar/`. Handles:
- Safe area adaptation for different devices (iOS/Android)
- Back navigation with configurable delta
- Menu button positioning
- Loading states
- Animated show/hide transitions

**Properties**: `title`, `background`, `color`, `back`, `loading`, `homeButton`, `animated`, `show`, `delta`

### App Configuration (app.json)

- Uses **Skyline renderer** with specific options:
  - `defaultDisplayBlock: true`
  - `defaultContentBox: true`
  - `tagNameStyleIsolation: "legacy"`
- **Lazy code loading**: "requiredComponents"
- Custom navigation style for all pages

### Page Structure

Pages are registered in `app.json` under the `pages` array. The first page in the array is the entry page.

## Code Patterns

### Component Definition
```javascript
Component({
  properties: { /* props */ },
  data: { /* state */ },
  lifetimes: { /* lifecycle methods */ },
  methods: { /* methods */ }
})
```

### Page Definition
```javascript
Page({
  data: { /* state */ },
  onLoad() { /* lifecycle */ }
})
```

### WeChat API Usage
Use `wx` global object for all WeChat APIs (e.g., `wx.getMenuButtonBoundingClientRect()`, `wx.navigateBack()`, `wx.getWindowInfo()`).
