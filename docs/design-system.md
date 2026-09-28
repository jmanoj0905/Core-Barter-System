# Design System

This document outlines the design language and tokens used in the Core Barter frontend.

## Philosophy

Brutalist-inspired with earthy/warm Material 3 colors. Sharp corners, strong typography hierarchy, and clear visual separation through borders.

## Colors

### Primary
- `--color-primary`: #000000
- `--color-on-primary`: #e2e2e2
- `--color-surface-container-low`: #f4f5e6
- `--color-surface-container`: #efefe0

### Semantic
- `--color-success`: #22c55e
- `--color-warning-mild`: #f59e0b
- `--color-warning-strong`: #f97316
- `--color-warning-severe`: #7c3aed
- `--color-error`: #ba1a1a

### Neutral
- `--color-surface`: #fafaeb (background)
- `--color-on-surface`: #1b1c14
- `--color-on-surface-variant`: #474747
- `--color-outline`: #777777
- `--color-outline-variant`: #c6c6c6

## Typography

- **Font Family**: Manrope (Google Fonts)
- **Headings**: Uppercase, tracking-tight, font-extrabold
- **Labels**: Uppercase, tracking-[0.2em], text-[10px], font-bold

## Components

### Headers
```jsx
<section className="mb-12 border-l-4 border-primary pl-8">
  <h1 className="text-4xl md:text-5xl font-extrabold tracking-tighter text-primary uppercase">
    Title
  </h1>
</section>
```

### Cards
```jsx
<div className="border border-outline-variant p-8">
  {/* content */}
</div>
```

### Buttons
- Primary: `bg-primary text-on-primary px-10 py-4 font-bold uppercase tracking-[0.3em] text-sm hover:opacity-90`
- Secondary: `border border-primary text-primary px-10 py-4 font-bold uppercase tracking-[0.3em] text-sm hover:bg-primary hover:text-on-primary`

### Form Inputs
```jsx
<input
  className="w-full bg-surface-container-low border border-outline px-4 py-4 focus:outline-none focus:border-primary transition-colors duration-200 text-primary font-medium"
/>
```

### Tab Bar
```jsx
<div className="grid grid-cols-4 border border-primary">
  {tabs.map(t => (
    <button
      className={`py-4 text-center font-bold uppercase tracking-widest text-xs ${
        tab === t ? 'bg-primary text-surface' : 'bg-transparent text-primary'
      }`}
    >
      {t}
    </button>
  ))}
</div>
```

## Animations

- `animate-blink`: For recording indicators
- `animate-pulse-ring`: For live audio visualization

## Layout

- Max content width: ~1200px
- Section spacing: mb-10 to mb-16
- Card padding: p-6 to p-8
- Grid gaps: gap-4
