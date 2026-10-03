-- Keep narrow numeric columns readable after Markdown table alignment.
function Table(tbl)
  local specs = tbl.colspecs
  local count = #specs
  local minimum = 0.11
  if count == 0 or count * minimum >= 1 then
    return tbl
  end
  local weights = {}
  for i = 1, count do
    weights[i] = 8
  end
  for _, body in ipairs(tbl.bodies) do
    for _, row in ipairs(body.body) do
      for i, cell in ipairs(row.cells) do
        local text = pandoc.utils.stringify(cell.contents)
        weights[i] = math.max(weights[i], math.min(32, utf8.len(text)))
      end
    end
  end
  local surplus = 0
  for _, weight in ipairs(weights) do
    surplus = surplus + weight - 8
  end
  local remaining = 1 - count * minimum
  for i, spec in ipairs(specs) do
    spec[2] = surplus > 0
      and minimum + remaining * (weights[i] - 8) / surplus
      or 1 / count
  end
  tbl.colspecs = specs
  return tbl
end
