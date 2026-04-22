import { FormEvent, useEffect, useMemo, useState } from 'react'
import './App.css'

type Point = {
  date: string
  value: number
}

type Citation = {
  source: string
  series_id: string
  description: string
  url: string
}

type MacroMetric = {
  metric_key: string
  label: string
  series_key: string
  series_id: string
  value: number
  unit: string
  date: string
  calculation: string
  description: string
}

type Message = {
  id: number
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  metrics?: MacroMetric[]
  warnings?: string[]
}

export type SeriesConfig = {
  key: string
  seriesId: string
  label: string
  shortLabel: string
  unit: string
  tone: string
  data: Point[]
  citation: Citation
}

type SeriesMetadataResponse = {
  key: string
  series_id: string
  description: string
}

type SeriesResponse = SeriesMetadataResponse & {
  points: Point[]
  citation: Citation
}

type AskResponse = {
  answer: string
  citations?: Citation[]
  metrics?: MacroMetric[]
  warnings?: string[]
  selectedSeries?: string | null
}

type SeriesUiConfig = {
  shortLabel: string
  unit: string
  tone: string
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, '') ?? ''
const DASHBOARD_LIMIT = 60

const SERIES_UI: Record<string, SeriesUiConfig> = {
  GDP: { shortLabel: 'GDP', unit: 'B USD', tone: 'teal' },
  REAL_GDP: { shortLabel: 'Real GDP', unit: 'B chained USD', tone: 'teal' },
  CPI: { shortLabel: 'CPI', unit: 'index', tone: 'amber' },
  EUROZONE_INFLATION: { shortLabel: 'Euro CPI', unit: '%', tone: 'blue' },
  UNEMPLOYMENT: { shortLabel: 'US Jobs', unit: '%', tone: 'rose' },
  JAPAN_UNEMPLOYMENT: { shortLabel: 'JP Jobs', unit: '%', tone: 'indigo' },
  FEDERAL_FUNDS_RATE: { shortLabel: 'Rates', unit: '%', tone: 'indigo' },
  USD_EUR: { shortLabel: 'FX', unit: 'USD', tone: 'blue' },
}

const fallbackTones = ['teal', 'amber', 'rose', 'indigo', 'blue']

const starterPrompts = [
  'What is the current US unemployment rate?',
  'Make a plot of US GDP growth over the past 10 years.',
  'How should I interpret the latest Fed funds rate?',
]

const unitDecimals = (unit: string) => {
  if (unit === '%') return 1
  if (unit === 'USD') return 2
  if (unit === 'USD per EUR') return 4
  return 1
}

export const formatValue = (series: SeriesConfig, value: number) => {
  if (series.unit === '%') return `${value.toFixed(1)}%`
  if (series.unit === 'USD') return value.toFixed(2)
  return value.toLocaleString(undefined, { maximumFractionDigits: unitDecimals(series.unit) })
}

export const formatMetricValue = (metric: MacroMetric) => {
  if (metric.unit === '%') return `${metric.value.toFixed(2)}%`
  if (metric.unit === 'USD per EUR') return metric.value.toFixed(4)
  return metric.value.toLocaleString(undefined, { maximumFractionDigits: 2 })
}

const errorMessage = async (response: Response) => {
  try {
    const data = (await response.json()) as { detail?: unknown }
    if (typeof data.detail === 'string') return data.detail
    if (
      data.detail &&
      typeof data.detail === 'object' &&
      'message' in data.detail &&
      typeof data.detail.message === 'string'
    ) {
      return data.detail.message
    }
  } catch {
    // The backend did not return JSON; fall back to the HTTP status.
  }

  return `Backend returned ${response.status}`
}

const fetchJson = async <T,>(path: string): Promise<T> => {
  const response = await fetch(`${API_BASE_URL}${path}`)
  if (!response.ok) {
    throw new Error(await errorMessage(response))
  }

  return response.json() as Promise<T>
}

const toSeriesConfig = (series: SeriesResponse, index: number): SeriesConfig => {
  const ui = SERIES_UI[series.key] ?? {
    shortLabel: series.key.replace(/_/g, ' '),
    unit: '',
    tone: fallbackTones[index % fallbackTones.length],
  }

  return {
    key: series.key,
    seriesId: series.series_id,
    label: series.description,
    shortLabel: ui.shortLabel,
    unit: ui.unit,
    tone: ui.tone,
    data: series.points,
    citation: series.citation,
  }
}

const loadDashboardSeries = async () => {
  const metadata = await fetchJson<SeriesMetadataResponse[]>('/series')
  const series = await Promise.all(
    metadata.map((item) => fetchJson<SeriesResponse>(`/series/${encodeURIComponent(item.key)}?limit=${DASHBOARD_LIMIT}`)),
  )

  return series.map(toSeriesConfig).filter((item) => item.data.length > 0)
}

const askBackend = async (query: string) => {
  const response = await fetch(`${API_BASE_URL}/ask`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query }),
  })

  if (!response.ok) {
    throw new Error(await errorMessage(response))
  }

  const data = (await response.json()) as AskResponse

  return {
    answer: data.answer,
    citations: data.citations ?? [],
    metrics: data.metrics ?? [],
    warnings: data.warnings ?? [],
    selectedSeries: data.selectedSeries ?? undefined,
  }
}

const Icon = ({ name }: { name: 'send' | 'chart' | 'source' }) => {
  const paths = {
    send: 'M3 11.5 20 4l-4.5 16-3.2-6.3L3 11.5Zm9 1 2.2 4.4 2.5-8.8L7.4 12.2 12 12.5Z',
    chart: 'M4 19h16v2H2V3h2v16Zm3-2V9h3v8H7Zm5 0V5h3v12h-3Zm5 0v-6h3v6h-3Z',
    source: 'M5 4h10l4 4v12H5V4Zm9 1.5V9h3.5L14 5.5ZM8 12v2h8v-2H8Zm0 4v2h8v-2H8Z',
  }

  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" focusable="false">
      <path d={paths[name]} />
    </svg>
  )
}

export const LineChart = ({ series }: { series: SeriesConfig }) => {
  const width = 720
  const height = 300
  const padding = 34
  const values = series.data.map((point) => point.value)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const range = max - min || 1
  const xStep = (width - padding * 2) / Math.max(1, series.data.length - 1)

  const points = series.data
    .map((point, index) => {
      const x = padding + index * xStep
      const y = height - padding - ((point.value - min) / range) * (height - padding * 2)
      return `${x},${y}`
    })
    .join(' ')

  const first = series.data[0]
  const last = series.data[series.data.length - 1]

  return (
    <div className="chart-shell">
      <div className="chart-header">
        <div>
          <p className="eyebrow">{series.seriesId}</p>
          <h2>{series.label}</h2>
        </div>
        <div className="latest-value">
          <span>{formatValue(series, last.value)}</span>
          <small>{last.date}</small>
        </div>
      </div>
      <svg className="line-chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={`${series.label} chart`}>
        <line x1={padding} y1={height - padding} x2={width - padding} y2={height - padding} />
        <line x1={padding} y1={padding} x2={padding} y2={height - padding} />
        <text x={padding} y={24}>{formatValue(series, max)}</text>
        <text x={padding} y={height - 8}>{formatValue(series, min)}</text>
        <polyline points={points} />
        <circle cx={width - padding} cy={height - padding - ((last.value - min) / range) * (height - padding * 2)} r="5" />
        <text x={padding} y={height - 10}>{first.date.slice(0, 4)}</text>
        <text x={width - padding - 38} y={height - 10}>{last.date.slice(0, 4)}</text>
      </svg>
    </div>
  )
}

function App() {
  const [selectedSeriesKey, setSelectedSeriesKey] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [messages, setMessages] = useState<Message[]>([])
  const [seriesList, setSeriesList] = useState<SeriesConfig[]>([])
  const [seriesError, setSeriesError] = useState<string | null>(null)
  const [isSeriesLoading, setIsSeriesLoading] = useState(true)
  const [isLoading, setIsLoading] = useState(false)

  useEffect(() => {
    let ignore = false

    const loadSeries = async () => {
      setIsSeriesLoading(true)
      setSeriesError(null)

      try {
        const loadedSeries = await loadDashboardSeries()
        if (ignore) return

        setSeriesList(loadedSeries)
        setSelectedSeriesKey((current) => {
          if (current && loadedSeries.some((series) => series.key === current)) return current
          return loadedSeries[0]?.key ?? null
        })
      } catch (error) {
        if (ignore) return

        const message = error instanceof Error ? error.message : 'Unable to load FRED series data.'
        setSeriesList([])
        setSelectedSeriesKey(null)
        setSeriesError(message)
      } finally {
        if (!ignore) setIsSeriesLoading(false)
      }
    }

    void loadSeries()

    return () => {
      ignore = true
    }
  }, [])

  const selectedSeries = useMemo(
    () => seriesList.find((series) => series.key === selectedSeriesKey) ?? seriesList[0] ?? null,
    [seriesList, selectedSeriesKey],
  )

  const handleSubmit = async (event: FormEvent) => {
    event.preventDefault()
    const trimmed = query.trim()
    if (!trimmed || isLoading) return

    const userMessage: Message = { id: Date.now(), role: 'user', content: trimmed }
    setMessages((current) => [...current, userMessage])
    setQuery('')
    setIsLoading(true)

    try {
      const result = await askBackend(trimmed)
      const resultSeriesKey = result.selectedSeries && seriesList.some((series) => series.key === result.selectedSeries)
        ? result.selectedSeries
        : null

      if (resultSeriesKey) setSelectedSeriesKey(resultSeriesKey)
      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: 'assistant',
          content: result.answer,
          citations: result.citations,
          metrics: result.metrics,
          warnings: result.warnings,
        },
      ])
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Backend request failed'
      setMessages((current) => [
        ...current,
        {
          id: Date.now() + 1,
          role: 'assistant',
          content: `Backend request failed: ${message}. Start the API server and try again.`,
        },
      ])
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="app">
      <section className="workspace">
        <header className="masthead">
          <div>
            <p className="eyebrow">AI macro specialist</p>
            <h1>Macro dashboard and analyst chat</h1>
          </div>
          <div className="source-pill">
            <Icon name="source" />
            FRED sourced
          </div>
        </header>

        <section className="indicator-grid" aria-label="Economic indicators">
          {isSeriesLoading ? (
            <div className="empty-state dashboard-state">
              <strong>Loading live FRED data.</strong>
              <span>The dashboard is requesting series from the backend API.</span>
            </div>
          ) : seriesError ? (
            <div className="empty-state dashboard-state">
              <strong>Unable to load live FRED data.</strong>
              <span>{seriesError}</span>
            </div>
          ) : (
            seriesList.map((series) => {
              const latest = series.data[series.data.length - 1]
              const previous = series.data[series.data.length - 5] ?? latest
              const change = latest.value - previous.value
              return (
                <button
                  className={`indicator-card ${series.key === selectedSeries?.key ? 'active' : ''}`}
                  data-tone={series.tone}
                  key={series.key}
                  onClick={() => setSelectedSeriesKey(series.key)}
                  type="button"
                >
                  <span>{series.shortLabel}</span>
                  <strong>{formatValue(series, latest.value)}</strong>
                  <small>{change >= 0 ? '+' : ''}{change.toFixed(unitDecimals(series.unit))} {series.unit}</small>
                </button>
              )
            })
          )}
        </section>

        {selectedSeries ? (
          <LineChart series={selectedSeries} />
        ) : (
          <div className="chart-shell">
            <div className="empty-state">
              <strong>No chart data loaded.</strong>
              <span>Start the backend API and refresh the page to fetch live FRED observations.</span>
            </div>
          </div>
        )}
      </section>

      <aside className="chat-panel" aria-label="Macro specialist chat">
        <div className="chat-title">
          <Icon name="chart" />
          <h2>Analyst chat</h2>
        </div>

        <div className="prompt-row">
          {starterPrompts.map((prompt) => (
            <button key={prompt} type="button" onClick={() => setQuery(prompt)}>
              {prompt}
            </button>
          ))}
        </div>

        <div className="message-list">
          {messages.length === 0 ? (
            <div className="empty-state">
              <strong>Ready for macro questions.</strong>
              <span>Responses include citations and update the chart context.</span>
            </div>
          ) : (
            messages.map((message) => (
              <article className={`message ${message.role}`} key={message.id}>
                <p>{message.content}</p>
                {message.citations?.length ? (
                  <div className="citation-list">
                    {message.citations.map((citation) => (
                      <a href={citation.url} key={citation.series_id} rel="noreferrer" target="_blank">
                        {citation.description}
                      </a>
                    ))}
                  </div>
                ) : null}
                {message.metrics?.length ? (
                  <div className="metric-list">
                    {message.metrics.map((metric) => (
                      <span key={`${metric.series_id}-${metric.metric_key}`}>
                        <strong>{metric.label}</strong>
                        {formatMetricValue(metric)} {metric.unit !== '%' ? metric.unit : ''}
                        <small>{metric.date}</small>
                      </span>
                    ))}
                  </div>
                ) : null}
                {message.warnings?.length ? (
                  <div className="warning-list">
                    {message.warnings.map((warning) => (
                      <span key={warning}>{warning}</span>
                    ))}
                  </div>
                ) : null}
              </article>
            ))
          )}
          {isLoading ? <div className="message assistant loading">Analyzing macro data...</div> : null}
        </div>

        <form className="chat-form" onSubmit={handleSubmit}>
          <input
            aria-label="Macro question"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask about GDP, inflation, labor, rates, or FX"
            value={query}
          />
          <button aria-label="Send question" disabled={isLoading || !query.trim()} type="submit">
            <Icon name="send" />
          </button>
        </form>
      </aside>
    </main>
  )
}

export default App
