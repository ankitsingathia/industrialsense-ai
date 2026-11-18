// IMS file names are timestamps: "2004.02.12.10.32.39".
export function parseStamp(stamp) {
  const [y, mo, d, h, mi, s] = stamp.split(".").map(Number);
  return new Date(y, mo - 1, d, h, mi, s);
}

export function at(start, hours) {
  return new Date(parseStamp(start).getTime() + hours * 3600 * 1000);
}

const fmt = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
  hour: "2-digit",
  minute: "2-digit",
});

export function formatAt(start, hours) {
  return fmt.format(at(start, hours));
}

export function formatHours(h) {
  if (h < 48) return `${h.toFixed(1)} h`;
  return `${(h / 24).toFixed(1)} days`;
}
