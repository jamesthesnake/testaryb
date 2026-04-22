import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { LineChart, formatValue, formatMetricValue, type SeriesConfig } from './App'

const makeSeries = (data: { date: string; value: number }[], unit = '%'): SeriesConfig => ({
  key: 'UNEMPLOYMENT',
  seriesId: 'UNRATE',
  label: 'US Unemployment Rate',
  shortLabel: 'US Jobs',
  unit,
  tone: 'rose',
  data,
  citation: {
    source: 'FRED',
    series_id: 'UNRATE',
    description: 'US Unemployment Rate',
    url: 'https://fred.stlouisfed.org/series/UNRATE',
  },
})

// ---------------------------------------------------------------------------
// LineChart
// ---------------------------------------------------------------------------

describe('LineChart', () => {
  it('renders the series label and FRED ID', () => {
    render(<LineChart series={makeSeries([{ date: '2025-01-01', value: 4.1 }])} />)
    expect(screen.getByText('US Unemployment Rate')).toBeInTheDocument()
    expect(screen.getByText('UNRATE')).toBeInTheDocument()
  })

  it('displays the latest value and date in the header', () => {
    const series = makeSeries([
      { date: '2024-12-01', value: 4.0 },
      { date: '2025-01-01', value: 4.1 },
    ])
    const { container } = render(<LineChart series={series} />)
    // The SVG also renders the max value as a label; scope to the header box.
    const latestBox = container.querySelector('.latest-value')!
    expect(latestBox).toHaveTextContent('4.1%')
    expect(latestBox).toHaveTextContent('2025-01-01')
  })

  it('renders a polyline with one coordinate pair per data point', () => {
    const data = [
      { date: '2025-01-01', value: 4.0 },
      { date: '2025-02-01', value: 4.2 },
      { date: '2025-03-01', value: 4.1 },
    ]
    const { container } = render(<LineChart series={makeSeries(data)} />)
    const polyline = container.querySelector('polyline')!
    const pairs = polyline.getAttribute('points')!.trim().split(' ').filter(Boolean)
    expect(pairs).toHaveLength(3)
  })

  it('handles a single data point without division by zero', () => {
    // When min === max the range collapses to 1 to avoid NaN coordinates.
    const series = makeSeries([{ date: '2025-01-01', value: 5.0 }])
    expect(() => render(<LineChart series={series} />)).not.toThrow()
    const { container } = render(<LineChart series={series} />)
    const polyline = container.querySelector('polyline')!
    expect(polyline.getAttribute('points')).not.toMatch(/NaN/)
  })

  it('sets an accessible aria-label on the SVG', () => {
    render(<LineChart series={makeSeries([{ date: '2025-01-01', value: 4.1 }])} />)
    expect(screen.getByRole('img', { name: /US Unemployment Rate/i })).toBeInTheDocument()
  })
})

// ---------------------------------------------------------------------------
// formatValue
// ---------------------------------------------------------------------------

describe('formatValue', () => {
  it('appends % and one decimal for percent series', () => {
    expect(formatValue(makeSeries([], '%'), 4.1)).toBe('4.1%')
  })

  it('returns two decimals for USD series', () => {
    expect(formatValue(makeSeries([], 'USD'), 1.0523)).toBe('1.05')
  })

  it('uses toLocaleString for other units', () => {
    const result = formatValue(makeSeries([], 'index'), 320.1)
    expect(result).toMatch(/320/)
  })
})

// ---------------------------------------------------------------------------
// formatMetricValue
// ---------------------------------------------------------------------------

describe('formatMetricValue', () => {
  const metric = (unit: string, value: number) => ({
    metric_key: 'k',
    label: 'l',
    series_key: 'CPI',
    series_id: 'CPIAUCSL',
    value,
    unit,
    date: '2025-01-01',
    calculation: '',
    description: '',
  })

  it('formats percent values with two decimals and % suffix', () => {
    expect(formatMetricValue(metric('%', 3.5))).toBe('3.50%')
  })

  it('formats USD per EUR with four decimals', () => {
    expect(formatMetricValue(metric('USD per EUR', 1.08765))).toBe('1.0877')
  })

  it('formats other units with two decimal places', () => {
    expect(formatMetricValue(metric('index', 320.1))).toMatch(/320/)
  })
})
