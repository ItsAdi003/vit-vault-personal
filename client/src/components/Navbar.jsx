import { useState, useEffect } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { motion } from 'framer-motion'
import { useAuth } from '../lib/AuthContext'

export default function Navbar({ variant = 'landing' }) {
  const isLanding = variant === 'landing'
  const { user, signOut } = useAuth()
  const navigate = useNavigate()

  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('theme') || 'dark';
  });

  useEffect(() => {
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  };

  const handleSignOut = async () => {
    await signOut()
    navigate('/')
  }

  return (
    <motion.header
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="sticky top-0 z-50 px-6 lg:px-20 py-4 glass border-b-0"
    >
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        <Link to="/" className="flex items-center gap-3">
          <div className="bg-[#3b3bed] p-2 rounded-lg">
            <span className="material-symbols-outlined text-white">auto_awesome_motion</span>
          </div>
          <div>
            <h2 className="text-xl font-bold tracking-tight text-slate-900 dark:text-white">VIT Media Vault</h2>
            {isLanding && (
              <p className="text-[10px] text-slate-500 tracking-wider uppercase hidden sm:block">
                Vidyalankar Institute of Technology
              </p>
            )}
          </div>
        </Link>

        {isLanding ? (
          <nav className="hidden md:flex items-center gap-10">
            <a className="text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-[#3b3bed] transition-colors" href="#home">Home</a>
            <Link className="text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-[#3b3bed] transition-colors" to="/gallery">Gallery</Link>
            <a className="text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-[#3b3bed] transition-colors" href="#events">Events</a>
            <a className="text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-[#3b3bed] transition-colors" href="#features">Features</a>
            <Link className="text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-[#3b3bed] transition-colors" to="/lab">AI Lab</Link>
          </nav>
        ) : (
          <nav className="hidden md:flex items-center gap-8">
            <Link className="text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-[#3b3bed] transition-colors" to="/">Dashboard</Link>
            <Link className="text-sm font-medium text-[#3b3bed]" to="/gallery">Events</Link>
            <a className="text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-[#3b3bed] transition-colors" href="#">Collections</a>
            <a className="text-sm font-medium text-slate-700 dark:text-slate-300 hover:text-[#3b3bed] transition-colors" href="#">Settings</a>
          </nav>
        )}

        <div className="flex items-center gap-4">
          <button
            onClick={toggleTheme}
            className="p-2 rounded-full hover:bg-slate-200 dark:hover:bg-white/10 text-slate-600 dark:text-slate-400 transition-colors flex items-center justify-center"
            title={theme === 'dark' ? "Switch to Light Mode" : "Switch to Dark Mode"}
          >
            <span className="material-symbols-outlined">
              {theme === 'dark' ? 'light_mode' : 'dark_mode'}
            </span>
          </button>
          
          {!user ? (
            <>
              <Link to="/login" className="text-sm font-medium px-4 py-2 hover:text-[#3b3bed] transition-colors text-slate-900 dark:text-white">Log In</Link>
              <Link to="/login" className="bg-[#3b3bed] hover:bg-[#3b3bed]/90 text-white text-sm font-bold px-6 py-2.5 rounded-lg transition-all glow-button">
                Sign Up
              </Link>
            </>
          ) : (
            <div className="flex items-center gap-3">
              <button className="p-2 rounded-full hover:bg-white/10 transition-colors hidden sm:block">
                <span className="material-symbols-outlined">search</span>
              </button>
              
              <div className="flex items-center gap-3 bg-slate-100 dark:bg-white/5 border border-slate-200 dark:border-white/10 rounded-full pl-3 pr-1 py-1">
                <span className="text-sm font-medium text-slate-800 dark:text-slate-300">
                  {user.email?.split('@')[0]}
                </span>
                <div className="size-8 rounded-full border-2 border-[#3b3bed]/50 bg-[#3b3bed] flex items-center justify-center text-xs font-bold text-white shadow-lg">
                  {user.email ? user.email[0].toUpperCase() : 'V'}
                </div>
              </div>

              <button 
                onClick={handleSignOut}
                className="p-2 ml-2 rounded-full hover:bg-red-500/10 hover:text-red-400 text-slate-400 transition-colors group"
                title="Sign Out"
              >
                <span className="material-symbols-outlined group-hover:scale-110 transition-transform">logout</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </motion.header>
  )
}
