// The one account allowed to manage every folder. Override per deployment with VITE_ADMIN_EMAIL in .env.local.
export const ADMIN_EMAIL = (import.meta.env.VITE_ADMIN_EMAIL || 'raunak.baweja@vit.edu.in').toLowerCase()

export const isAdminEmail = (email) => !!email && email.toLowerCase() === ADMIN_EMAIL
